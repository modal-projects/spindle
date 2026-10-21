"""Async lifecycle around the Rust scheduler; no policy logic lives here."""

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import httpx
from spindle_router import Decision, Request, Snapshot
from spindle_router import Router as Scheduler

from spindle.telemetry.http_trace import HTTPTrace

from .router_transport import RoutedResponse

log = logging.getLogger(__name__)
SnapshotReader = Callable[[], Awaitable[list[dict]]]


class Router:
    def __init__(
        self,
        discover: SnapshotReader,
        *,
        policy,
        capacity,
        rates,
        close,
        balance_absolute=4,
        balance_relative=1.25,
        cache_threshold=0.5,
    ):
        self.scheduler = Scheduler(
            policy,
            capacity,
            rates["prefill_tokens_per_second"],
            rates["decode_tokens_per_second"],
            rates["gpu_lora_seconds"],
            rates["registration_seconds"],
            balance_absolute,
            balance_relative,
            cache_threshold,
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
                    snapshot = await asyncio.to_thread(
                        Snapshot,
                        json.dumps(rows),
                    )
                    self.scheduler.install_snapshot(snapshot)
                except Exception:
                    # A discovery failure must not kill the poller.
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
        affinity_started = None
        while True:
            self.last_used = time.monotonic()
            if self.poller is None or self.poller.done():
                self.poller = asyncio.create_task(self._poll())
            self.changed.clear()
            elapsed = (
                0.0 if affinity_started is None else loop.time() - affinity_started
            )
            decision = self.scheduler.select_destination(request, request_id, elapsed)
            if (
                decision.reason in {"pending_prefix", "cached_capacity"}
                and affinity_started is None
            ):
                affinity_started = loop.time()
            if decision.status == "route":
                return decision
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Routing deadline exceeded: {decision.status}")
            # Jitter also limits correlated retries across independent workers.
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self.changed.wait(),
                    min(remaining, decision.retry_after, random.uniform(0.25, 0.75)),
                )

    async def dispatch(
        self,
        prepared,
        *,
        request_id,
        gateway,
        body,
        headers,
        client,
        deadline,
        trace_attrs=None,
    ):
        began = asyncio.get_running_loop().time()
        decision = await self.select_destination(
            prepared, request_id, deadline=deadline
        )
        routing = dict(
            policy=self.policy,
            reason=decision.reason,
            available_replicas=decision.available_replicas,
            cached_tokens=decision.cached_tokens,
            best_cached_tokens=decision.best_cached_tokens,
            backlog_seconds=decision.backlog_seconds,
            prefill_seconds=decision.prefill_seconds,
            adapter_seconds=decision.adapter_seconds,
            wait_s=asyncio.get_running_loop().time() - began,
        )
        status, failed = None, True
        try:
            headers = {
                k: v
                for k, v in headers.items()
                if k.lower()
                not in {
                    "modal-session-id",
                    "modal-routing-affinity-key",
                    "modal-flash-upstream",
                }
            }
            headers["Modal-Flash-Upstream"] = decision.upstream
            request = client.build_request(
                "POST",
                gateway + "/generate",
                json=body,
                headers=headers,
                extensions={"trace": HTTPTrace(trace_attrs).record}
                if trace_attrs is not None
                else {},
                timeout=max(0.001, deadline - asyncio.get_running_loop().time()),
            )
            request.headers.pop("Modal-Session-ID", None)
            request.headers.pop("Modal-Routing-Affinity-Key", None)
            response = await client.send(request)
            status = response.status_code
            actual = response.headers.get("x-spindle-replica")
            failed = status >= 500 or bool(actual and actual != decision.replica)
            if actual and actual != decision.replica:
                raise httpx.TransportError("Worker identity mismatch")
            return RoutedResponse(response, decision.replica, routing)
        except asyncio.CancelledError:
            failed = False
            raise
        finally:
            self.complete(request_id, status=status, failed=failed)

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
