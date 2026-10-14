import asyncio
import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts.benchmarks.lora_routing.cohort_prefill import prepare_cohort
from spindle.inference.lora_routing import Replica
from spindle.inference.routing_domains import DomainRouter
from spindle.inference.routing_work import WorkRates, token_prefixes


def setup(samples=8):
    router = DomainRouter(
        AsyncMock(),
        rates=WorkRates(1000, 100, 0.1, 1, 4),
        balanced=True,
        stable=True,
        prefill_once=True,
    )
    adapter = "m/weight_v000004"
    router.replicas = {
        h: Replica(
            h,
            upstream=h,
            work=dict(
                age_seconds=0,
                prefixes=frozenset(),
                prefill_tokens=0,
                decode_tokens=0,
                pending_adapters=[],
                request_ids=frozenset(),
                gpu_adapters=[],
                cpu_adapters=[adapter],
            ),
        )
        for h in ("a", "b")
    }
    router.updated = time.monotonic()
    router.refresh = AsyncMock()
    task = dict(
        request_id="group",
        sampling_session_id="session",
        model_id="m",
        publish_version=4,
        payload=dict(
            prompt={"chunks": [{"type": "encoded_text", "tokens": list(range(1024))}]},
            num_samples=samples,
            cache_affinity_key="affinity",
            sampling_params={"max_tokens": 128},
        ),
    )
    return router, task


def response(request, version=4):
    return httpx.Response(
        200,
        headers={"x-spindle-replica": request.headers["Modal-Flash-Upstream"]},
        json={
            "output_ids": [42],
            "meta_info": {
                "weight_version_start": version,
                "weight_version_end": version,
                "cached_tokens": 0,
            },
        },
    )


def test_preparation_reserves_whole_group_and_seeds_each_planned_host_once():
    async def run():
        router, task = setup()
        seen, records = [], []

        async def send(request):
            assert len(router.reservations) == 8
            body = json.loads(request.content)
            assert body["input_ids"] == list(range(960))
            assert body["sampling_params"]["max_new_tokens"] == 1
            assert body["weight_version"]["exact_version"] == 4
            seen.append(request.headers["Modal-Flash-Upstream"])
            return response(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            await prepare_cohort(task, "https://pool", client, router, records)
        assert sorted(seen) == ["a", "b"]
        assert len(records) == 2
        assert not router.reservations and not router.host_reservations
        assert set(router.group_routes.values()) == {"a", "b"}

    asyncio.run(run())


def test_warm_boundary_needs_no_auxiliary_generation():
    async def run():
        router, task = setup()
        for replica in router.replicas.values():
            replica.work["prefixes"] = frozenset(
                h for h, _ in token_prefixes("m/weight_v000004", list(range(1024)))
            )
        client = AsyncMock()
        await prepare_cohort(task, "https://pool", client, router, [])
        client.post.assert_not_called()
        assert not router.reservations

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["transport", "version", "cancel"])
def test_failed_or_cancelled_preparation_releases_every_reservation(failure):
    async def run():
        router, task = setup(samples=2)
        entered = asyncio.Event()

        async def send(request):
            entered.set()
            if failure == "transport":
                raise httpx.ReadError("injected")
            if failure == "version":
                return response(request, version=3)
            await asyncio.Future()

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            pending = asyncio.create_task(
                prepare_cohort(task, "https://pool", client, router, [])
            )
            await entered.wait()
            if failure == "cancel":
                pending.cancel()
            expected = {
                "transport": httpx.ReadError,
                "version": AssertionError,
                "cancel": asyncio.CancelledError,
            }[failure]
            with pytest.raises(expected):
                await pending
        assert not router.reservations and not router.host_reservations
        assert not any(router.local.values())

    asyncio.run(run())
