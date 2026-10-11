"""Observe replicas through the authenticated Flash gateway, off the hot path."""

import asyncio
import time

import httpx
from stitch.pools.modal_flash import list_flash_containers_async

from spindle.inference.lora_routing import LoraRouter, Replica

from .fft_pool import proxy_auth_headers


def create_router(
    app_name: str, gateway: str, *, target: int, slots: int
) -> LoraRouter:
    containers = []
    discovered = 0.0

    async def discover():
        nonlocal containers, discovered
        now = time.monotonic()
        if now - discovered > 5 or not containers:
            containers = await list_flash_containers_async(app_name, "Server")
            discovered = now
        async with httpx.AsyncClient(
            headers=proxy_auth_headers(), timeout=2, trust_env=False
        ) as client:

            async def inspect(container):
                if not container.host:
                    return None
                upstream = (
                    f"{container.host}:{container.port}"
                    if container.port
                    else container.host
                )
                try:
                    response = await client.get(
                        gateway.rstrip("/") + "/routing-state",
                        headers={"Modal-Flash-Upstream": upstream},
                    )
                    response.raise_for_status()
                    state = response.json()
                    # Never attribute an observation to the wrong replica, even
                    # if an older proxy ignores the exact-upstream header.
                    if state["replica"] != container.task_id:
                        return None
                    return Replica(
                        container.task_id,
                        state["active"],
                        state["adapters"],
                        tuple(state["recent"]),
                        upstream,
                    )
                except (httpx.HTTPError, ValueError, KeyError):
                    return None

            observations = await asyncio.gather(*(inspect(c) for c in containers))
        return [r for r in observations if r is not None]

    return LoraRouter(discover, target=target, slots=slots)
