"""Async lifecycle around the Rust scheduler; no policy logic lives here."""

import asyncio
from contextlib import suppress
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable

from spindle_router import Decision, Request, Router as Scheduler

log = logging.getLogger(__name__)
SnapshotReader = Callable[[], Awaitable[list[dict]]]


class Router:
    def __init__(self, discover: SnapshotReader, *, policy, capacity, rates, close):
        self.scheduler = Scheduler(
            policy,
            capacity,
            rates["prefill_tokens_per_second"],
            rates["decode_tokens_per_second"],
            rates["gpu_lora_seconds"],
            rates["registration_seconds"],
        )
        self.policy = policy
        self.discover = discover
        self.close_transport = close
        self.poller = None
        self.changed = asyncio.Event()
        self.last_used = time.monotonic()

    def prepare_request(self, adapter, affinity, prompt, output_tokens):
        # Construct once per sampling group, including its version-scoped hashes.
        return Request(adapter, affinity, prompt, output_tokens)

    async def _poll(self):
        try:
            while self.scheduler.pending or time.monotonic() - self.last_used < 60:
                try:
                    rows = await asyncio.wait_for(self.discover(), timeout=3)
                    self.scheduler.update_snapshot(json.dumps(rows))
                except Exception:  # Telemetry/discovery failure must not kill the poller.
                    log.warning(
                        "Routing observations unavailable; waiting for a fresh snapshot"
                    )
                self.changed.set()
                await asyncio.sleep(0.5)
        finally:
            self.changed.set()

    async def select_destination(
        self, request: Request, request_id: str, *, deadline: float
    ) -> Decision:
        """Wait locally for admission; never turn pool saturation into Flash fallback."""
        loop = asyncio.get_running_loop()
        while True:
            self.last_used = time.monotonic()
            if self.poller is None or self.poller.done():
                self.poller = asyncio.create_task(self._poll())
            self.changed.clear()
            decision = self.scheduler.select_destination(request, request_id)
            if decision.status == "route":
                return decision
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Routing deadline exceeded: {decision.status}")
            # Jitter also limits correlated retries across independent workers.
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self.changed.wait(), min(remaining, random.uniform(0.25, 0.75))
                )

    def complete(self, request_id, *, status=None, failed=False):
        outcome = "busy" if status in (429, 503) else "failed" if failed else "ok"
        self.scheduler.complete(request_id, outcome)
        self.changed.set()

    async def aclose(self):
        if self.poller is not None:
            self.poller.cancel()
            with suppress(asyncio.CancelledError):
                await self.poller
        await self.close_transport()
