"""Discover placement hints through the authenticated Flash gateway."""

import asyncio
import time

import httpx
from stitch.pools.modal_flash import list_flash_containers_async

from spindle.inference.routing import Engine, InferenceRouter, SchedulingPolicy

from .fft_pool import proxy_auth_headers


def create_router(app_name, gateway, *, policy: SchedulingPolicy):
    containers, discovered = [], 0.0
    client = httpx.AsyncClient(headers=proxy_auth_headers(), timeout=2, trust_env=False)

    async def discover():
        nonlocal containers, discovered
        now = time.monotonic()
        if now - discovered > 5 or not containers:
            containers = await list_flash_containers_async(app_name, "Server")
            discovered = now

        async def inspect(container):
            if not container.host:
                return None
            upstream = (
                f"{container.host}:{container.port}"
                if container.port
                else container.host
            )
            try:
                began = time.monotonic()
                response = await client.get(
                    gateway.rstrip("/") + "/routing-state",
                    headers={"Modal-Flash-Upstream": upstream},
                )
                response.raise_for_status()
                state = response.json()
                if state["replica"] != container.task_id:
                    return None
                observed = state["age_seconds"] is not None
                return Engine(
                    container.task_id,
                    upstream,
                    state["active"],
                    frozenset(state["request_ids"]),
                    tuple(state["cpu_adapters"]) if observed else (),
                    tuple(state["pending_adapters"]) if observed else (),
                    state["age_seconds"] + time.monotonic() - began
                    if observed
                    else float("inf"),
                )
            except (httpx.HTTPError, ValueError, KeyError):
                return None

        observations = await asyncio.gather(*(inspect(c) for c in containers))
        return [r for r in observations if r is not None]

    return InferenceRouter(discover, policy, close=client.aclose)
