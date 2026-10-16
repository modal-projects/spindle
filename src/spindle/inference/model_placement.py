"""Balanced model placement with load-based overflow (the model-spill policy)."""

from collections import OrderedDict
from functools import lru_cache
import time

import xxhash

from .routing import PoolSnapshot, RoutingRequest


@lru_cache(maxsize=1024)
def _ranking(identity: str, hosts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            hosts,
            key=lambda host: xxhash.xxh3_64_intdigest(
                (identity + "\0" + host).encode()
            ),
            reverse=True,
        )
    )


class ModelPlacement:
    def __init__(self, *, capacity: int):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Model routing capacity must be a positive integer")
        self.capacity = capacity
        self.prefix_destinations = OrderedDict()
        self.group_routes = OrderedDict()
        self.host_seen, self.adapter_seen = {}, {}
        self.catalog_key, self.primary_hosts = None, {}

    def _placement_topology(self, adapter, pool):
        now = time.monotonic()
        for host, engine in pool.engines.items():
            self.host_seen[host] = pool.observed_at
            for name in engine.cpu_adapters + engine.pending_adapters:
                self.adapter_seen[name] = pool.observed_at
        self.adapter_seen[adapter] = now
        # A missed observation or temporary rejection must not remap other models.
        self.host_seen = {h: t for h, t in self.host_seen.items() if now - t < 10}
        self.adapter_seen = {a: t for a, t in self.adapter_seen.items() if now - t < 10}
        self.adapter_seen.pop("base", None)
        return tuple(sorted(self.host_seen))

    def _primary_placement(self, adapter, hosts, pool):
        catalog = {adapter, *self.adapter_seen}
        for engine in pool.engines.values():
            catalog.update(engine.cpu_adapters)
            catalog.update(engine.pending_adapters)
        catalog.discard("base")
        # Versions of one training model share placement, never cache identity.
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

    def select_engine(self, request: RoutingRequest, pool: PoolSnapshot) -> str | None:
        hosts = tuple(sorted(pool.loads))
        if not hosts:
            return None
        adapter = request.adapter
        key = xxhash.xxh3_128_hexdigest((adapter + "\0" + request.affinity).encode())
        chunk = request.sample_index // self.capacity
        count = min(self.capacity, request.group_size - chunk * self.capacity)
        group_key = (
            (adapter, request.group_id, chunk) if request.group_id is not None else None
        )
        planned = self.group_routes.get(group_key)
        if planned in hosts:
            hosts = (planned,)
        loads = {host: pool.loads[host] for host in _ranking(key, hosts)}
        if len(loads) > 1:
            ranking = self._primary_placement(
                adapter, self._placement_topology(adapter, pool), pool
            )
            primary = next(h for h in ranking if h in loads)
            # Preserve the evaluated policy's 25% relative imbalance allowance,
            # plus one sibling chunk. This is a heuristic, not an admission cap.
            limit = 1.25 * sum(loads.values()) / len(loads) + count
            if loads[primary] <= limit:
                loads = {primary: loads[primary]}
        chosen = min(loads, key=loads.__getitem__)
        previous = self.prefix_destinations.get(key)
        if previous in loads and loads[previous] <= loads[chosen]:
            chosen = previous
        self._remember(self.prefix_destinations, key, chosen)
        if group_key is not None:
            self._remember(self.group_routes, group_key, chosen)
        return chosen

    @staticmethod
    def _remember(routes, key, chosen):
        routes[key] = chosen
        routes.move_to_end(key)
        if len(routes) > 8192:
            routes.popitem(last=False)
