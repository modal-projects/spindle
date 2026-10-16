import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
import xxhash
from spindle_router import Request, Router as Scheduler

from spindle.inference.router import Router
from spindle.inference.sampling import sample_task

RATES = dict(
    prefill_tokens_per_second=1000.0,
    decode_tokens_per_second=100.0,
    gpu_lora_seconds=0.1,
    registration_seconds=1.0,
)


def replica(name="a", active=0):
    return dict(
        replica=name,
        upstream=name,
        active=active,
        work=dict(
            age_seconds=0,
            prefixes=[],
            cpu_adapters=["m/weight_v000007"],
            gpu_adapters=["m/weight_v000007"],
            pending_adapters=[],
            request_ids=[],
            prefill_tokens=0,
            decode_tokens=0,
        ),
    )


def test_rust_hashes_match_engine_prefix_protocol():
    from array import array

    digest = xxhash.xxh3_128(b"m/weight_v000007\0")
    digest.update(array("q", [42] * 64).tobytes())
    a, b = replica("a"), replica("b")
    b["work"]["prefixes"] = [digest.hexdigest()]
    scheduler = Scheduler("joint", 32, 1000.0, 100.0, 0.1, 1.0)
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request("m/weight_v000007", "group", [42] * 100, 1)
    selected = scheduler.select_destination(q, "r")
    assert selected.replica == "b" and selected.cached_tokens == 64


def test_saturation_waits_then_admits_without_flash():
    async def run():
        state = [replica(active=1)]
        router = Router(
            AsyncMock(side_effect=lambda: state),
            policy="joint",
            capacity=1,
            rates=RATES,
            close=AsyncMock(),
        )
        q = router.prepare_request("m/weight_v000007", "group", [1], 1)
        pending = asyncio.create_task(
            router.select_destination(
                q, "r", deadline=asyncio.get_running_loop().time() + 2
            )
        )
        await asyncio.sleep(0.05)
        assert not pending.done()
        state[0]["active"] = 0
        assert (await pending).replica == "a"
        router.complete("r")
        assert router.scheduler.pending == 0
        await router.aclose()

    asyncio.run(run())


def test_missing_observations_timeout_without_reservation():
    async def run():
        router = Router(
            AsyncMock(return_value=[]),
            policy="joint",
            capacity=1,
            rates=RATES,
            close=AsyncMock(),
        )
        q = router.prepare_request("m/weight_v000007", "group", [1], 1)
        with pytest.raises(TimeoutError):
            await router.select_destination(
                q, "r", deadline=asyncio.get_running_loop().time() + 0.02
            )
        assert router.scheduler.pending == 0
        await router.aclose()

    asyncio.run(run())


def test_sampling_retries_preserve_router_and_release_reservations(monkeypatch):
    import spindle.inference.sampling as sampling

    monkeypatch.setattr(sampling, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(sampling, "RETRY_MAX_DELAY_SECONDS", 0.01)

    async def run():
        router = Router(
            AsyncMock(return_value=[replica()]),
            policy="joint",
            capacity=32,
            rates=RATES,
            close=AsyncMock(),
        )
        requests = []

        def handle(req):
            requests.append(req)
            assert req.headers["Modal-Flash-Upstream"] == "a"
            assert "Modal-Session-ID" not in req.headers
            if len(requests) == 1:
                return httpx.Response(503)
            return httpx.Response(
                200,
                headers={"x-spindle-replica": "a"},
                json={
                    "meta_info": {
                        "finish_reason": "stop",
                        "output_token_logprobs": [[-0.1, 5]],
                        "weight_version_start": 7,
                        "weight_version_end": 7,
                    }
                },
            )

        task = dict(
            request_id="group",
            sampling_session_id="s",
            model_id="m",
            publish_version=7,
            payload=dict(
                prompt=dict(chunks=[dict(type="encoded_text", tokens=[1])]),
                sampling_params=dict(max_tokens=1),
                num_samples=2,
            ),
        )
        result = await sample_task(
            task, "http://gateway", router=router, transport=httpx.MockTransport(handle)
        )
        assert len(result["sequences"]) == 2
        assert len(requests) == 3 and router.scheduler.pending == 0
        await router.aclose()

    asyncio.run(run())


def test_cancelling_generation_releases_reservation():
    async def run():
        router = Router(
            AsyncMock(return_value=[replica()]),
            policy="joint",
            capacity=1,
            rates=RATES,
            close=AsyncMock(),
        )
        started = asyncio.Event()

        async def handle(req):
            started.set()
            await asyncio.Event().wait()

        task = dict(
            request_id="group",
            sampling_session_id="s",
            model_id="m",
            publish_version=7,
            payload=dict(
                prompt=dict(chunks=[dict(type="encoded_text", tokens=[1])]),
                sampling_params=dict(max_tokens=1),
            ),
        )
        pending = asyncio.create_task(
            sample_task(
                task,
                "http://gateway",
                router=router,
                transport=httpx.MockTransport(handle),
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert router.scheduler.pending == 0
        await router.aclose()

    asyncio.run(run())
