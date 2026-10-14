"""Observe replicas through the authenticated Flash gateway, off the hot path."""

import asyncio
import time

import httpx
from stitch.pools.modal_flash import list_flash_containers_async

from spindle.inference.lora_routing import LoraRouter, Replica
from spindle.inference.routing_work import WorkRates, WorkRouter
from spindle.inference.routing_domains import DomainRouter
from spindle.inference.routing_simple import SimpleRouter
from spindle.inference.routing_capacity import CapacityRouter

from .fft_pool import proxy_auth_headers


def create_router(
    app_name: str,
    gateway: str,
    *,
    target: int,
    slots: int,
    work_rates: dict | None = None,
    adapter_costs: bool = True,
    domains: bool = False,
    balanced_domains: bool = False,
    stable_domains: bool = False,
    prefill_once: bool = False,
    simple_policy: str | None = None,
    capacity_aware: bool = False,
    capacity_tie_seed: str | None = None,
) -> LoraRouter:
    containers = []
    discovered = 0.0
    # Reuse connections across snapshots. A new TLS client per poll made refresh
    # unnecessarily expensive under high request concurrency.
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
                # Never attribute an observation to the wrong replica, even
                # if an older proxy ignores the exact-upstream header.
                if state["replica"] != container.task_id:
                    return None
                work = state.get("work")
                if work is not None:
                    work["admission_supported"] = state.get(
                        "admission_supported", False
                    )
                    work["age_seconds"] += time.monotonic() - began
                    work["prefixes"] = frozenset(work["prefixes"])
                    work["request_ids"] = frozenset(work["request_ids"])
                return Replica(
                    container.task_id,
                    state["active"],
                    state["adapters"],
                    tuple(state["recent"]),
                    upstream,
                    frozenset(state.get("prefixes", ())),
                    work,
                )
            except (httpx.HTTPError, ValueError, KeyError):
                return None

        observations = await asyncio.gather(*(inspect(c) for c in containers))
        return [r for r in observations if r is not None]

    async def resolve_affinity(affinity):
        response = await client.get(
            gateway.rstrip("/") + "/routing-state",
            headers={"Modal-Session-ID": affinity},
        )
        response.raise_for_status()
        return response.json()["replica"]

    router_type = (
        (DomainRouter if domains else WorkRouter)
        if work_rates is not None
        else LoraRouter
    )
    extra = (
        dict(
            rates=WorkRates(**work_rates),
            adapter_costs=adapter_costs,
            prefill_once=prefill_once,
        )
        if work_rates is not None
        else {}
    )
    if domains:
        extra["balanced"] = balanced_domains
        extra["stable"] = stable_domains
    if simple_policy is not None:
        if work_rates is None or domains:
            raise ValueError("Simple controls require rates and no domain flags")
        router_type = SimpleRouter
        extra["policy"] = simple_policy
    if capacity_aware:
        if work_rates is None or domains or simple_policy is not None:
            raise ValueError("Capacity routing requires rates and no placement flags")
        router_type = CapacityRouter
        extra["tie_seed"] = capacity_tie_seed
    return router_type(
        discover,
        target=target,
        slots=slots,
        close=client.aclose,
        resolve_affinity=resolve_affinity,
        **extra,
    )
