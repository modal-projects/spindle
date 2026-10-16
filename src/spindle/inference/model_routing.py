"""Model placement with bounded overflow, using frontend-local observations.

This is the model-spill policy evaluated in the serving study. Placement is
version-independent; reservations and conversation locality remain exact-version
scoped. Counts are scheduling hints, not a distributed admission semaphore.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import logging
import time

import xxhash


@lru_cache(maxsize=1024)
def _ranking(identity, hosts):
    return tuple(
        sorted(
            hosts,
            key=lambda host: xxhash.xxh3_64_intdigest(
                (identity + "\0" + host).encode()
            ),
            reverse=True,
        )
    )


@dataclass
class Replica:
    replica_id: str
    upstream: str
    active: int
    request_ids: frozenset[str]
    cpu_adapters: tuple[str, ...]
    pending_adapters: tuple[str, ...]
    age_seconds: float


class ModelRouter:
    def __init__(self, discover, *, capacity, close=None):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Model routing capacity must be a positive integer")
        self.discover, self.capacity, self._close = discover, capacity, close
        self.replicas = {}
        self.reservations = {}
        self.host_reservations = {}
        self.failed_until = {}
        self.prefix_homes = OrderedDict()
        self.group_routes = OrderedDict()
        self.host_seen, self.adapter_seen = {}, {}
        self.catalog_key, self.primary_hosts = None, {}
        self.updated = self.next_refresh = 0.0
        self.refresh_interval = 1.0
        self.stale_after = 10.0
        self.pending = self.poller = None

    async def aclose(self):
        tasks = [t for t in (self.pending, self.poller) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._close is not None:
            await self._close()

    async def _poll_in_flight(self):
        while True:
            await asyncio.sleep(self.refresh_interval)
            if not self.reservations:
                return
            await self.refresh()

    async def refresh(self):
        if self.poller is None or self.poller.done():
            self.poller = asyncio.create_task(self._poll_in_flight())
        now = time.monotonic()
        if now >= self.next_refresh and (self.pending is None or self.pending.done()):
            self.next_refresh = now + self.refresh_interval
            self.pending = asyncio.create_task(self._refresh())
        # Fresh selections do not wait. Coalesce stale bursts behind one
        # bounded observation request; failure falls back to Flash affinity.
        if self.pending is not None and (
            not self.updated
            or now - self.updated > self.stale_after
            or not any(
                r.age_seconds + now - self.updated < 3 for r in self.replicas.values()
            )
        ):
            await asyncio.shield(self.pending)

    async def _refresh(self):
        try:
            replicas = await asyncio.wait_for(self.discover(), timeout=3.0)
        except Exception:
            logging.getLogger(__name__).warning(
                "Model placement discovery failed", exc_info=True
            )
            return
        self.replicas = {r.replica_id: r for r in replicas}
        self.updated = time.monotonic()
        self.failed_until = {
            h: t for h, t in self.failed_until.items() if t > self.updated
        }

    def _placement_topology(self, adapter):
        now = time.monotonic()
        for host, replica in self.replicas.items():
            self.host_seen[host] = self.updated
            for name in replica.cpu_adapters + replica.pending_adapters:
                self.adapter_seen[name] = self.updated
        self.adapter_seen[adapter] = now
        self.host_seen = {
            h: t for h, t in self.host_seen.items() if now - t < self.stale_after
        }
        self.adapter_seen = {
            a: t for a, t in self.adapter_seen.items() if now - t < self.stale_after
        }
        self.adapter_seen.pop("base", None)
        return tuple(sorted(self.host_seen))

    def _primary_placement(self, adapter, hosts):
        catalog = {adapter, *self.adapter_seen}
        for replica in self.replicas.values():
            catalog.update(replica.cpu_adapters)
            catalog.update(replica.pending_adapters)
        catalog.discard("base")
        catalog = {a.rsplit("/", 1)[0] for a in catalog}
        key = (hosts, frozenset(catalog))
        if key != self.catalog_key:
            assigned = dict.fromkeys(hosts, 0)
            self.primary_hosts = {}
            for name in sorted(catalog):
                host = min(_ranking(name, hosts), key=assigned.__getitem__)
                assigned[host] += 1
                self.primary_hosts[name] = host
            self.catalog_key = key
        primary = self.primary_hosts[adapter.rsplit("/", 1)[0]]
        return (primary,) + tuple(h for h in _ranking(adapter, hosts) if h != primary)

    def select_destination(
        self,
        adapter,
        affinity,
        *,
        reservation_id,
        group_id=None,
        group_size=1,
        sample_index=0,
        details=None,
    ):
        now = time.monotonic()
        if now - self.updated > self.stale_after:
            return None
        hosts = tuple(
            sorted(
                h
                for h, r in self.replicas.items()
                if self.failed_until.get(h, 0) <= now
                and r.age_seconds + now - self.updated < 3
            )
        )
        if not hosts:
            return None
        key = xxhash.xxh3_128_hexdigest((adapter + "\0" + affinity).encode())
        chunk = sample_index // self.capacity
        count = min(self.capacity, group_size - chunk * self.capacity)
        group_key = (adapter, group_id, chunk) if group_id is not None else None
        planned = self.group_routes.get(group_key)
        if planned in hosts:
            hosts = (planned,)
        loads = {}
        for host in _ranking(key, hosts):
            replica = self.replicas[host]
            loads[host] = replica.active + sum(
                rid not in replica.request_ids
                for rid in self.host_reservations.get(host, {})
            )
        if len(loads) > 1:
            ranking = self._primary_placement(
                adapter, self._placement_topology(adapter)
            )
            primary = next(h for h in ranking if h in loads)
            limit = 1.25 * sum(loads.values()) / len(loads) + count
            if loads[primary] <= limit:
                loads = {primary: loads[primary]}
        chosen = min(loads, key=loads.__getitem__)
        previous = self.prefix_homes.get(key)
        if previous in loads and loads[previous] <= loads[chosen]:
            chosen = previous
        self.prefix_homes[key] = chosen
        self.prefix_homes.move_to_end(key)
        if len(self.prefix_homes) > 8192:
            self.prefix_homes.popitem(last=False)
        if group_key is not None:
            self.group_routes[group_key] = chosen
            self.group_routes.move_to_end(group_key)
            if len(self.group_routes) > 8192:
                self.group_routes.popitem(last=False)
        self.reservations[reservation_id] = chosen
        self.host_reservations.setdefault(chosen, set()).add(reservation_id)
        if details is not None:
            details.update(
                observed_and_reserved_requests=loads[chosen],
                selection_seconds=time.monotonic() - now,
            )
        return chosen

    def release(self, host, *, reservation_id, failed=False):
        reserved = self.reservations.pop(reservation_id, None)
        if reserved is not None:
            requests = self.host_reservations[reserved]
            requests.remove(reservation_id)
            if not requests:
                del self.host_reservations[reserved]
        if host is not None and failed:
            self.failed_until[host] = time.monotonic() + 2 * self.refresh_interval
            self.next_refresh = 0.0
