"""Pool-local routing lifecycle; scheduling policies do no I/O."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import logging
import time
from typing import Protocol


@dataclass
class Engine:
    replica_id: str
    upstream: str
    active: int
    request_ids: frozenset[str]
    cpu_adapters: tuple[str, ...]
    pending_adapters: tuple[str, ...]
    age_seconds: float


@dataclass(frozen=True)
class RoutingRequest:
    adapter: str
    affinity: str
    request_id: str
    group_id: str | None = None
    group_size: int = 1
    sample_index: int = 0


@dataclass(frozen=True)
class PoolSnapshot:
    # All observations retain catalog/topology continuity through a busy replica.
    engines: Mapping[str, Engine]
    # Only fresh, non-failed engines are eligible. Loads include unseen local work.
    loads: Mapping[str, int]
    observed_at: float


class SchedulingPolicy(Protocol):
    def select_engine(self, request: RoutingRequest, pool: PoolSnapshot) -> str | None:
        """Return an engine ID in pool.loads, or None to use native routing."""
        ...


class InferenceRouter:
    """One instance per pool per sampling worker, used on a single event loop.

    Selection and reservation are synchronous. Refreshes coalesce into one task;
    only cold/stale callers await it, and cancellation cannot cancel that task.
    Reservations are local hints, not distributed admission control.
    """

    def __init__(
        self,
        discover: Callable[[], Awaitable[list[Engine]]],
        policy: SchedulingPolicy,
        *,
        close: Callable[[], Awaitable[None]] | None = None,
    ):
        self.discover, self.policy, self._close = discover, policy, close
        self.engines: dict[str, Engine] = {}
        self.reservations: dict[str, str] = {}
        self.host_reservations: dict[str, set[str]] = {}
        self.failed_until: dict[str, float] = {}
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
        if self.pending is not None and (
            not self.updated
            or now - self.updated > self.stale_after
            or not any(
                e.age_seconds + now - self.updated < 3 for e in self.engines.values()
            )
        ):
            await asyncio.shield(self.pending)

    async def _refresh(self):
        try:
            engines = await asyncio.wait_for(self.discover(), timeout=3.0)
        except Exception:
            logging.getLogger(__name__).warning(
                "Engine discovery failed", exc_info=True
            )
            return
        self.engines = {e.replica_id: e for e in engines}
        self.updated = time.monotonic()
        self.failed_until = {
            h: t for h, t in self.failed_until.items() if t > self.updated
        }

    def select_engine(self, request: RoutingRequest, *, details=None) -> Engine | None:
        now = time.monotonic()
        if now - self.updated > self.stale_after:
            return None
        loads = {
            host: engine.active
            + sum(
                rid not in engine.request_ids
                for rid in self.host_reservations.get(host, ())
            )
            for host, engine in self.engines.items()
            if self.failed_until.get(host, 0) <= now
            and engine.age_seconds + now - self.updated < 3
        }
        if not loads:
            return None
        chosen = self.policy.select_engine(
            request, PoolSnapshot(self.engines, loads, self.updated)
        )
        if chosen is None:
            return None
        if chosen not in loads:
            raise ValueError("Scheduling policy selected an ineligible engine")
        self.reservations[request.request_id] = chosen
        self.host_reservations.setdefault(chosen, set()).add(request.request_id)
        if details is not None:
            details.update(
                observed_and_reserved_requests=loads[chosen],
                selection_seconds=time.monotonic() - now,
            )
        return self.engines[chosen]

    def finish_request(self, request_id: str, *, failed: bool = False):
        host = self.reservations.pop(request_id, None)
        if host is None:
            return
        requests = self.host_reservations[host]
        requests.remove(request_id)
        if not requests:
            del self.host_reservations[host]
        if failed:
            self.failed_until[host] = time.monotonic() + 2 * self.refresh_interval
            self.next_refresh = 0.0
