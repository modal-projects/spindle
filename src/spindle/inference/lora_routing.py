"""Replica-local observations and bounded-load rendezvous routing for LoRA.

Every sampling worker keeps its own snapshot. No shared lock, queue, or RPC is
needed to choose a warm route. Observations are hints, never weight guarantees.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache

import xxhash


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


class LoraRouter:
    def __init__(
        self,
        discover: Callable[[], Awaitable[list[Replica]]],
        *,
        target: int = 16,
        slots: int = 8,
        refresh_interval: float = 1.0,
        stale_after: float = 10.0,
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

    async def refresh(self):
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

    def select_destination(self, model: str, adapter: str, affinity: str) -> str | None:
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
        # An adapter may occupy additional replicas when its home becomes busy.
        # Share the additive allowance across the pool so a small workload
        # can use idle GPUs. At high load the relative bound allows locality
        # without queueing behind an overloaded home.
        limit = min(loads.values()) * 1.25 + max(1, self.target / len(hosts))
        eligible = [h for h in rank if loads[h] < limit]

        def warm(host):
            replica = self.replicas[host]
            return (
                adapter in self.local.get(host, {})
                or adapter in replica.adapters
                or adapter in replica.recent[-self.slots :]
            )

        warm_hosts = [h for h in eligible if warm(h)]
        if warm_hosts:
            # Stable conversation affinity within the warm set. At high load
            # the eligibility bound still permits spillover.
            chosen = _ranking(affinity, tuple(sorted(warm_hosts)))[0]
        else:
            # Avoid introducing a new adapter into a full active GPU working set.
            available = [
                h
                for h in eligible
                if len(set(self.replicas[h].adapters) | set(self.local.get(h, {})))
                < self.slots
            ]
            chosen = (available or eligible)[0]
        self.local.setdefault(chosen, Counter())[adapter] += 1
        return chosen

    def release(self, upstream: str | None, adapter: str, *, failed: bool = False):
        if upstream is None:
            return
        counts = self.local[upstream]
        counts[adapter] -= 1
        if counts[adapter] == 0:
            del counts[adapter]
        if failed:
            self.failed_until[upstream] = time.monotonic() + self.refresh_interval * 2
            self.next_refresh = 0.0
