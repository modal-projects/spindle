import asyncio
from collections import Counter
import json
import time
from unittest.mock import AsyncMock

import httpx

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_simple import SimpleRouter
from spindle.inference.routing_work import WorkRates, token_prefixes
from spindle.inference.sampling import sample_task


def router():
    result = SimpleRouter(
        AsyncMock(),
        policy="guarded-locality",
        rates=WorkRates(8770, 1085, 0.064, 2.2, 36),
    )
    result.refresh = AsyncMock()
    result.updated = time.monotonic()
    result.replicas = {
        f"h{i}": Replica(
            f"h{i}",
            upstream=f"h{i}",
            work=dict(
                admission_supported=True,
                age_seconds=0,
                prefixes=frozenset(
                    h for h, _ in token_prefixes("m/weight_v000007", list(range(8192)))
                )
                if i == 0
                else frozenset(),
                prefill_tokens=0,
                decode_tokens=0,
                pending_adapters=[],
                request_ids=frozenset(),
                cpu_adapters=["m/weight_v000007"],
                gpu_adapters=["m/weight_v000007"] if i == 0 else [],
            ),
        )
        for i in range(8)
    }
    return result


def task(frontend, samples):
    return dict(
        request_id=f"f{frontend}",
        sampling_session_id=f"s{frontend}",
        model_id="m",
        publish_version=7,
        payload=dict(
            num_samples=samples,
            prompt={"chunks": [{"type": "encoded_text", "tokens": list(range(8192))}]},
            sampling_params={"max_tokens": 1},
        ),
    )


def completed(host, guarded):
    return httpx.Response(
        200,
        headers={
            "x-spindle-replica": host,
            "x-spindle-routing-admitted": "1" if guarded else "0",
        },
        json={
            "meta_info": {
                "output_token_logprobs": [[-0.1, 5]],
                "weight_version_start": 7,
                "weight_version_end": 7,
            }
        },
    )


def test_correlated_warm_burst_uses_all_replicas_without_overadmitting():
    async def run():
        active = {f"h{i}": set() for i in range(8)}
        peaks = Counter()
        release = asyncio.Event()
        rejections = 0

        async def handle(request):
            nonlocal rejections
            host = request.headers["Modal-Flash-Upstream"]
            limit = int(request.headers["X-Spindle-Routing-Limit"])
            if len(active[host]) >= limit:
                rejections += 1
                return httpx.Response(
                    429,
                    headers={
                        "x-spindle-replica": host,
                        "x-spindle-routing-rejected": "1",
                    },
                    json={
                        "active": len(active[host]),
                        "request_ids": list(active[host]),
                    },
                )
            rid = json.loads(request.content)["rid"]
            active[host].add(rid)
            peaks[host] = max(peaks[host], len(active[host]))
            if sum(map(len, active.values())) == 288:
                release.set()
            await release.wait()
            active[host].remove(rid)
            return completed(host, True)

        routers = [router() for _ in range(8)]
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        sample_task(
                            task(i, 36),
                            "http://pool",
                            lora_router=r,
                            transport=httpx.MockTransport(handle),
                        )
                        for i, r in enumerate(routers)
                    )
                ),
                10,
            )
            assert all(len(output["sequences"]) == 36 for output in outputs)
            assert rejections > 0
            assert dict(peaks) == {f"h{i}": 36 for i in range(8)}
            assert not any(active.values())
            assert all(not r.reservations for r in routers)
        finally:
            await asyncio.gather(*(r.aclose() for r in routers))

    asyncio.run(run())


def test_all_busy_probes_are_bounded_then_request_queues_without_guard():
    async def run():
        r = router()
        attempts = []

        async def handle(request):
            host = request.headers["Modal-Flash-Upstream"]
            guarded = "X-Spindle-Routing-Limit" in request.headers
            attempts.append((host, guarded))
            if guarded:
                return httpx.Response(
                    429,
                    headers={
                        "x-spindle-replica": host,
                        "x-spindle-routing-rejected": "1",
                    },
                    json={"active": 36, "request_ids": []},
                )
            return completed(host, False)

        result = await sample_task(
            task(0, 1),
            "http://pool",
            lora_router=r,
            transport=httpx.MockTransport(handle),
        )
        assert result["sequences"][0]["tokens"] == [5]
        assert len(attempts) == 9
        assert len({host for host, guarded in attempts if guarded}) == 8
        assert attempts[-1][1] is False
        assert not r.reservations and not r.failed_until
        await r.aclose()

    asyncio.run(run())


def test_guarded_policy_requires_advertised_admission_support():
    r = router()
    del r.replicas["h0"].work["admission_supported"]
    assert (
        r.select_destination("m", "m/weight_v000007", "key", reservation_id="x") is None
    )
    assert not r.reservations
