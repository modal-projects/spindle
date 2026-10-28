"""Lifecycle/control-plane wrapper; Rust owns admission, retries and worker HTTP."""

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import httpx
from spindle_router import NativeRuntime


@dataclass
class RoutedResponse:
    response: httpx.Response
    replica: str | None
    routing: dict[str, Any]


class NativeRouter:
    def __init__(self, discover, *, gateway, headers, **settings):
        self.discover = discover
        self.gateway = gateway.rstrip("/")
        self.runtime = NativeRuntime(
            json.dumps(dict(gateway=self.gateway, headers=headers, **settings))
        )
        self.poller = None

    async def _discover(self):
        while True:
            try:
                workers = await asyncio.wait_for(self.discover(), timeout=5)
                self.runtime.set_workers(json.dumps(workers))
            except Exception:
                # Failed discovery must not permanently kill address refresh.
                # Rust independently enforces fresh observations and deadlines.
                pass
            await asyncio.sleep(5)

    async def dispatch(self, adapter, affinity, *, request_id, body, headers, deadline):
        if self.poller is None or self.poller.done():
            self.poller = asyncio.create_task(self._discover())
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Routing deadline expired")
        (
            status,
            response_headers,
            content,
            replica,
            routing,
        ) = await self.runtime.generate(
            json.dumps(
                dict(
                    request_id=request_id,
                    adapter=adapter,
                    affinity=affinity,
                    body=body,
                    headers=headers,
                    timeout_seconds=remaining,
                )
            )
        )
        response = httpx.Response(
            status,
            headers=response_headers,
            content=content,
            request=httpx.Request("POST", self.gateway + "/generate"),
        )
        if routing is None:
            # Sampling already handles retryable statuses within its deadline.
            return RoutedResponse(
                response, None, dict(reason="router_error", wait_s=0.0)
            )
        return RoutedResponse(response, replica, json.loads(routing))

    async def aclose(self):
        if self.poller is not None:
            self.poller.cancel()
            with suppress(asyncio.CancelledError):
                await self.poller
        self.runtime.close()
