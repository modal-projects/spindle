"""Replica-local observations and bounded-load rendezvous routing for LoRA.

Every sampling worker keeps its own snapshot. No shared lock, queue, or RPC is
needed to choose a warm route. Observations are hints, never weight guarantees.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache

import xxhash


def prefix_key(adapter: str, affinity: str) -> str:
    return xxhash.xxh3_128_hexdigest((adapter + "\0" + affinity).encode())


@lru_cache(maxsize=1024)
def _ranking(model: str, upstreams: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            upstreams,
            key=lambda host: xxhash.xxh3_64_intdigest((model + "\0" + host).encode()),
            reverse=True,
        )
    )


@dataclass
class Replica:
    replica_id: str
    active: int = 0
    adapters: dict[str, int] = field(default_factory=dict)
    recent: tuple[str, ...] = ()
    upstream: str = ""
    prefixes: frozenset[str] = frozenset()
    work: dict | None = None


class LoraRouter:
    def __init__(
        self,
        discover: Callable[[], Awaitable[list[Replica]]],
        *,
        target: int = 16,
        slots: int = 8,
        refresh_interval: float = 1.0,
        stale_after: float = 10.0,
        close: Callable[[], Awaitable[None]] | None = None,
        resolve_affinity: Callable[[str], Awaitable[str]] | None = None,
    ):
        if target < 1 or slots < 1:
            raise ValueError("routing target and adapter slots must be positive")
        self.discover = discover
        self.target, self.slots = target, slots
        self.refresh_interval, self.stale_after = refresh_interval, stale_after
        self.replicas: dict[str, Replica] = {}
        self.local: dict[str, Counter] = {}
        self.observed_local: dict[str, int] = {}
        self.failed_until: dict[str, float] = {}
        self.updated = 0.0
        self.next_refresh = 0.0
        self.pending: asyncio.Task | None = None
        self.poller: asyncio.Task | None = None
        self._close = close
        self.resolve_affinity = resolve_affinity
        self.prefix_lookups: dict[str, asyncio.Task] = {}
        self.lookup_count = self.lookup_failures = 0
        self.lookup_seconds = self.lookup_max_seconds = 0.0
        # A prefix keeps its first destination even when other replicas become
        # warm. Overflow does not rewrite its home on every generation.
        self.prefix_homes: OrderedDict[str, str] = OrderedDict()

    async def aclose(self):
        tasks = [t for t in (self.pending, self.poller) if t is not None]
        tasks.extend(self.prefix_lookups.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._close is not None:
            await self._close()

    def _remember_prefix(self, key: str, home: str):
        self.prefix_homes[key] = home
        self.prefix_homes.move_to_end(key)
        if len(self.prefix_homes) > 8192:
            self.prefix_homes.popitem(last=False)

    async def prepare_prefix(self, adapter: str, affinity: str, *, prompt=()):
        """Resolve an unseen prefix once through native Flash affinity.

        Concurrent callers share only this prefix's lookup. Warm prefixes do no
        network work here; destination selection and reservations remain local.
        """
        if self.resolve_affinity is None or not self.replicas:
            return
        key = prefix_key(adapter, affinity)
        if self.prefix_homes.get(key) in self.replicas:
            return
        known = tuple(h for h, r in self.replicas.items() if key in r.prefixes)
        if known:
            self._remember_prefix(key, _ranking(key, known)[0])
            return

        async def lookup():
            began = time.monotonic()
            try:
                home = await asyncio.wait_for(self.resolve_affinity(affinity), 2.0)
            except Exception:
                self.lookup_failures += 1
                return
            finally:
                elapsed = time.monotonic() - began
                self.lookup_count += 1
                self.lookup_seconds += elapsed
                self.lookup_max_seconds = max(self.lookup_max_seconds, elapsed)
            if home in self.replicas:
                self._remember_prefix(key, home)
            else:
                self.lookup_failures += 1

        if key not in self.prefix_lookups:
            task = asyncio.create_task(lookup())
            self.prefix_lookups[key] = task
            task.add_done_callback(lambda _: self.prefix_lookups.pop(key, None))
        await asyncio.shield(self.prefix_lookups[key])

    async def _poll_in_flight(self):
        while True:
            await asyncio.sleep(self.refresh_interval)
            if not any(self.local.values()):
                return
            await self.refresh()

    async def refresh(self):
        if self.poller is None or self.poller.done():
            self.poller = asyncio.create_task(self._poll_in_flight())
        now = time.monotonic()
        if now >= self.next_refresh and (self.pending is None or self.pending.done()):
            self.next_refresh = now + self.refresh_interval
            self.pending = asyncio.create_task(self._refresh())
        # Fresh snapshots never block. After an idle interval, coalesce the new
        # burst behind one bounded refresh instead of routing the entire burst
        # naively before observations arrive. Failed discovery falls back to Flash.
        if (
            not self.updated or now - self.updated > self.stale_after
        ) and self.pending is not None:
            await asyncio.shield(self.pending)

    async def _refresh(self):
        local_at_observation = {h: c.total() for h, c in self.local.items()}
        try:
            # Bound even the initial control-plane lookup; generation can fall
            # back to Flash if discovery or a replica is slow.
            replicas = await asyncio.wait_for(self.discover(), timeout=3.0)
        except Exception:
            logging.getLogger(__name__).warning(
                "LoRA replica discovery failed", exc_info=True
            )
            return
        self.replicas = {r.replica_id: r for r in replicas}
        self.observed_local = local_at_observation
        self.updated = time.monotonic()
        self.failed_until = {
            h: t for h, t in self.failed_until.items() if t > self.updated
        }
        self.local = {h: c for h, c in self.local.items() if c or h in self.replicas}

    def note_busy(self, host: str, active: int, request_ids):
        """Incorporate direct admission feedback until the next observation."""
        if host in self.replicas:
            replica = self.replicas[host]
            replica.active = active
            if replica.work is not None:
                replica.work["request_ids"] = frozenset(request_ids)

    def select_destination(
        self, model: str, adapter: str, affinity: str, **kwargs
    ) -> str | None:
        """Choose a replica and reserve its load until release() is called."""
        now = time.monotonic()
        if now - self.updated > self.stale_after:
            return None
        hosts = tuple(
            sorted(h for h in self.replicas if self.failed_until.get(h, 0) <= now)
        )
        if not hosts:
            return None
        rank = _ranking(model, hosts)
        loads = {}
        for host in hosts:
            local = self.local.get(host, Counter()).total()
            # Add arrivals since observation. max(reported, local) alone would
            # hide every new local request until local exceeded a busy replica's
            # entire reported load. Existing reservations must not count twice.
            arrivals = max(0, local - self.observed_local.get(host, 0))
            loads[host] = max(self.replicas[host].active + arrivals, local)
        # Allow one configured target of excess backlog for locality. A bound
        # exactly at the pool mean scatters every sample in a group once all
        # replicas are busy. The additive budget still spills an overloaded
        # prefix, and remains meaningful at both low and high pool load.
        # This bounds our observation, not distributed admission.
        limit = min(loads.values()) + self.target
        eligible = [h for h in rank if loads[h] < limit]

        key = prefix_key(adapter, affinity)
        home = self.prefix_homes.get(key)
        if home not in hosts:
            # Replica observations let another frontend recover prefix locality.
            # These are recent-use hints, not assertions about resident KV.
            known = tuple(h for h in hosts if key in self.replicas[h].prefixes)
            home = _ranking(key, known)[0] if known else None
        if home is None and self.resolve_affinity is not None:
            return None
        if home in eligible:
            chosen = home
        else:
            cached = [h for h in eligible if key in self.replicas[h].prefixes]
            working = {
                h: set(self.replicas[h].adapters) | set(self.local.get(h, {}))
                for h in eligible
            }
            # Start from KV affinity. Concentrating every new prefix on a
            # model home can manufacture hot queues when adapter capacity is
            # plentiful. Redirect cold placement only when its working set is
            # full; an established prefix continues to win over adapter warmth.
            choices = [h for h in _ranking(key, hosts) if h in eligible][:2]
            preferred = min(choices, key=loads.__getitem__)
            fits = adapter in working[preferred] or len(working[preferred]) < self.slots
            warm = [
                h
                for h in eligible
                if adapter in working[h]
                or adapter in self.replicas[h].recent[-self.slots :]
            ]
            room = [h for h in eligible if len(working[h]) < self.slots]
            candidates = cached or (
                warm or room or [preferred]
                if home is not None
                else [preferred]
                if fits
                else warm or room or [preferred]
            )
            chosen = min(candidates, key=loads.__getitem__)

        if home is None:
            home = chosen
        self._remember_prefix(key, home)
        self.local.setdefault(chosen, Counter())[adapter] += 1
        return chosen

    def release(
        self,
        upstream: str | None,
        adapter: str,
        *,
        failed: bool = False,
        reservation_id=None,
    ):
        if upstream is None:
            return
        counts = self.local[upstream]
        counts[adapter] -= 1
        if counts[adapter] == 0:
            del counts[adapter]
        if failed:
            self.failed_until[upstream] = time.monotonic() + self.refresh_interval * 2
            self.next_refresh = 0.0
