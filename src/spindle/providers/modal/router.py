"""Modal discovery and transport addresses. Scheduling is provider-independent."""

import asyncio
import time

import httpx
from stitch.pools.modal_flash import list_flash_containers_async

from spindle.inference.router import Router
from .fft_pool import proxy_auth_headers


def create_router(
    app_name,
    gateway,
    *,
    policy,
    capacity,
    rates,
    balance_absolute=4,
    balance_relative=1.25,
):
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
                if state["replica"] != container.task_id or state["work"] is None:
                    return None
                state["work"]["age_seconds"] += time.monotonic() - began
                return dict(
                    replica=state["replica"],
                    upstream=upstream,
                    active=state["active"],
                    work=state["work"],
                )
            except (httpx.HTTPError, ValueError, KeyError):
                return None

        return [
            row
            for row in await asyncio.gather(*(inspect(c) for c in containers))
            if row is not None
        ]

    return Router(
        discover,
        policy=policy,
        capacity=capacity,
        rates=rates,
        close=client.aclose,
        balance_absolute=balance_absolute,
        balance_relative=balance_relative,
    )
