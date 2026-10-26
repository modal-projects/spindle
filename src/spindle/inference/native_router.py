"""Lifecycle/control-plane wrapper; no per-request PyO3 calls or cache snapshots."""

import asyncio
import json
import secrets
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import quote

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
        self.token = secrets.token_urlsafe(32)
        self.runtime = NativeRuntime(
            json.dumps(
                dict(
                    gateway=self.gateway,
                    token=self.token,
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
        self.client = httpx.AsyncClient(
            base_url=self.runtime.address,
            headers={"Authorization": f"Bearer {self.token}"},
            trust_env=False,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
        self.poller = None

    def prepare_request(self, adapter, affinity, prompt, output_tokens):
        return RoutingKey(adapter, affinity, secrets.token_hex(16))

    async def _discover(self):
        while True:
            try:
                workers = await asyncio.wait_for(self.discover(), timeout=5)
                response = await self.client.put("/workers", json=workers, timeout=5)
                response.raise_for_status()
            except (httpx.HTTPError, TimeoutError):
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
        try:
            response = await self.client.post(
                "/generate",
                json=dict(
                    request_id=request_id,
                    group=prepared.group,
                    adapter=prepared.adapter,
                    affinity=prepared.affinity,
                    body=body,
                    headers=headers,
                    timeout_seconds=remaining,
                ),
                timeout=remaining + 1,
            )
            if response.is_error:
                # Sampling already handles retryable statuses within its deadline.
                # Preserve that path rather than leaking HTTPStatusError through RPC.
                return RoutedResponse(
                    response,
                    None,
                    dict(policy=self.policy, reason="router_error", wait_s=0.0),
                )
            row = response.json()
            upstream_response = httpx.Response(
                row["status"],
                headers=row["headers"],
                content=row["body"].encode(),
                request=httpx.Request("POST", gateway + "/generate"),
            )
            return RoutedResponse(upstream_response, row["replica"], row["routing"])
        except asyncio.CancelledError:
            with suppress(httpx.HTTPError):
                await asyncio.shield(
                    self.client.delete(
                        "/requests/" + quote(request_id, safe=""), timeout=2
                    )
                )
            raise

    async def aclose(self):
        if self.poller is not None:
            self.poller.cancel()
            with suppress(asyncio.CancelledError):
                await self.poller
        await asyncio.to_thread(self.runtime.close)
        await self.client.aclose()
