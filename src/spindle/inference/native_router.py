"""Lifecycle/control-plane wrapper; Rust owns admission, retries and worker HTTP."""

import asyncio
import json
import secrets
from contextlib import suppress
from dataclasses import dataclass

import httpx
from spindle_router import NativeRuntime

from .router_transport import RoutedResponse


@dataclass(frozen=True)
class RoutingKey:
    adapter: str
    affinity: str
    group: str


class NativeRouter:
    def __init__(
        self,
        discover,
        *,
        gateway,
        headers,
        policy,
        capacity,
        rates,
        balance_absolute=4,
        balance_relative=1.25,
        cache_threshold=0.5,
    ):
        self.policy = policy
        self.discover = discover
        self.gateway = gateway.rstrip("/")
        self.runtime = NativeRuntime(
            json.dumps(
                dict(
                    gateway=self.gateway,
                    headers=headers,
                    policy=policy,
                    capacity=capacity,
                    rates=rates,
                    balance_absolute=balance_absolute,
                    balance_relative=balance_relative,
                    cache_threshold=cache_threshold,
                )
            )
        )
        self.poller = None

    def prepare_request(self, adapter, affinity, prompt, output_tokens):
        return RoutingKey(adapter, affinity, secrets.token_hex(16))

    async def _discover(self):
        while True:
            try:
                workers = await asyncio.wait_for(self.discover(), timeout=5)
                self.runtime.set_workers(json.dumps(workers))
            except TimeoutError:
                # Failed discovery must not permanently kill address refresh.
                # Rust independently enforces fresh observations and deadlines.
                pass
            await asyncio.sleep(5)

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
        if gateway.rstrip("/") != self.gateway:
            raise ValueError("Native router belongs to a different inference pool")
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
                    group=prepared.group,
                    adapter=prepared.adapter,
                    affinity=prepared.affinity,
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
            request=httpx.Request("POST", gateway + "/generate"),
        )
        if routing is None:
            # Sampling already handles retryable statuses within its deadline.
            return RoutedResponse(
                response,
                None,
                dict(policy=self.policy, reason="router_error", wait_s=0.0),
            )
        return RoutedResponse(response, replica, json.loads(routing))

    async def aclose(self):
        if self.poller is not None:
            self.poller.cancel()
            with suppress(asyncio.CancelledError):
                await self.poller
        self.runtime.close()
