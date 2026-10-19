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
    b["work"]["prefix_nodes"] = [("m/weight_v000007", digest.hexdigest(), None, True)]
    scheduler = Scheduler("cache_wait", 32, 1000.0, 100.0, 0.1, 1.0)
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request("m/weight_v000007", "group", [42] * 100, 1)
    selected = scheduler.select_destination(q, "r")
    assert selected.replica == "b" and selected.cached_tokens == 64


def test_saturation_waits_then_admits_without_flash():
    policy = "cache_wait"

    async def run():
        state = [replica(active=1)]
        router = Router(
            AsyncMock(side_effect=lambda: state),
            policy=policy,
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
            policy="cache_wait",
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
            policy="cache_wait",
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
            policy="cache_wait",
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


def catalog(adapter, tokens, reusable=True):
    from array import array

    digest = xxhash.xxh3_128((adapter + "\0").encode())
    rows, parent = [], None
    for end in range(64, len(tokens), 64):
        digest.update(array("q", tokens[end - 64 : end]).tobytes())
        key = digest.hexdigest()
        rows.append((adapter, key, parent, reusable))
        parent = key
    return rows


def test_radix_load_gate_overrides_warmth_and_sibling_preference():
    a, b = replica("a"), replica("b")
    b["work"]["gpu_adapters"] = []
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request("m/weight_v000007", "group", [42] * 1024, 1)
    assert scheduler.select_destination(q, "first").replica == "a"
    scheduler.complete("first", "ok")
    a["active"] = 5
    scheduler.update_snapshot(json.dumps([a, b]))
    assert scheduler.select_destination(q, "overflow").replica == "b"


def test_radix_exact_version_eviction_and_no_optimistic_credit():
    policy = "cache_wait"
    a, b = replica("a"), replica("b")
    adapter = "m/weight_v000007"
    b["work"]["prefix_nodes"] = catalog(adapter, [42] * 1024)
    scheduler = Scheduler(policy, 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request(adapter, "group", [42] * 1024, 1)
    hit = scheduler.select_destination(q, "hit")
    assert (hit.replica, hit.cached_tokens) == ("b", 960)
    scheduler.complete("hit", "ok")
    new = Request("m/weight_v000008", "group", [42] * 1024, 1)
    assert scheduler.select_destination(new, "new-version").cached_tokens == 0
    b["work"]["prefix_nodes"] = []
    scheduler.update_snapshot(json.dumps([a, b]))
    assert scheduler.select_destination(q, "evicted").cached_tokens == 0


def test_invalid_snapshot_keeps_previous_tree():
    adapter = "m/weight_v000007"
    a = replica()
    a["work"]["prefix_nodes"] = catalog(adapter, [42] * 1024)
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a]))
    a["work"]["prefix_nodes"][0] = (adapter, "01", "02", True)
    with pytest.raises(ValueError):
        scheduler.update_snapshot(json.dumps([a]))
    q = Request(adapter, "group", [42] * 1024, 1)
    assert scheduler.select_destination(q, "valid").cached_tokens == 960


@pytest.mark.parametrize("cached_length,expected", [(512, "idle"), (576, "cache")])
def test_cache_wait_requires_more_than_half_the_prompt(cached_length, expected):
    adapter = "m/weight_v000007"
    a, b = replica("idle"), replica("cache", active=1)
    b["work"]["prefix_nodes"] = catalog(adapter, [42] * (cached_length + 1))
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request(adapter, "group", [42] * 1024, 1)
    assert scheduler.select_destination(q, "r").replica == expected


def test_cache_wait_prefers_long_prefix_over_hot_adapter():
    adapter = "m/weight_v000007"
    a, b = replica("hot"), replica("cache", active=2)
    b["work"]["gpu_adapters"] = []
    b["work"]["prefix_nodes"] = catalog(adapter, [42] * 8192)
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request(adapter, "group", [42] * 8192, 1)
    assert scheduler.select_destination(q, "r").replica == "cache"


def test_cache_wait_load_fallback_spreads_group_despite_warm_adapter():
    adapter = "m/weight_v000007"
    a, b = replica("hot"), replica("cold")
    b["work"]["gpu_adapters"] = []
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request(adapter, "group", [42] * 1024, 1)
    assert scheduler.select_destination(q, "first").replica == "hot"
    assert scheduler.select_destination(q, "second", 2.01).replica == "cold"


def test_cache_wait_equal_prefixes_prefer_hot_adapter():
    adapter = "m/weight_v000007"
    a, b = replica("cold"), replica("hot")
    a["work"]["gpu_adapters"] = []
    for row in [a, b]:
        row["work"]["prefix_nodes"] = catalog(adapter, [42] * 1024)
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    q = Request(adapter, "group", [42] * 1024, 1)
    assert scheduler.select_destination(q, "r").replica == "hot"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_cache_wait_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="cache_threshold"):
        Scheduler("cache_wait", 32, *RATES.values(), cache_threshold=threshold)


def wait_router(a_active=32, b_active=31, cached=8192):
    adapter = "m/weight_v000007"
    a, b = replica("cache", a_active), replica("other", b_active)
    a["work"]["prefix_nodes"] = catalog(adapter, [42] * cached)
    scheduler = Scheduler("cache_wait", 32, *RATES.values())
    scheduler.update_snapshot(json.dumps([a, b]))
    return scheduler, Request(adapter, "group", [42] * 8192, 1), a, b


def test_full_cached_replica_waits_then_admits_when_free():
    scheduler, q, a, b = wait_router()
    d = scheduler.select_destination(q, "r")
    assert (d.status, d.reason, d.available_replicas) == ("wait", "cached_capacity", 1)
    assert d.retry_after == 2 and d.best_cached_tokens == 8128
    assert scheduler.pending == 0
    a["active"] = 31
    scheduler.update_snapshot(json.dumps([a, b]))
    d = scheduler.select_destination(q, "r", 0.1)
    assert (d.replica, d.cached_tokens) == ("cache", 8128)


def test_cache_wait_budget_expires_without_reset_on_new_snapshot():
    scheduler, q, a, b = wait_router()
    assert scheduler.select_destination(q, "r").status == "wait"
    scheduler.update_snapshot(json.dumps([a, b]))
    d = scheduler.select_destination(q, "r", 2.01)
    assert (d.status, d.replica, d.cached_tokens) == ("route", "other", 0)


def test_cache_wait_budget_scales_with_saved_prefill():
    scheduler, q, a, b = wait_router(cached=1024)
    q = Request("m/weight_v000007", "group", [42] * 1024, 1)
    assert scheduler.select_destination(q, "r").retry_after == pytest.approx(0.96)
    assert scheduler.select_destination(q, "r", 0.97).replica == "other"


def test_overloaded_warm_replica_spills_immediately():
    scheduler, q, _, _ = wait_router(b_active=0)
    assert scheduler.select_destination(q, "r").replica == "other"


def test_all_full_is_capacity_wait_not_affinity_wait():
    scheduler, q, _, _ = wait_router(b_active=32)
    d = scheduler.select_destination(q, "r")
    assert (d.status, d.reason, scheduler.pending) == ("wait", "capacity", 0)


def test_pending_group_waits_for_observed_prefix_without_optimistic_hit():
    scheduler, q, a, b = wait_router(0, 0, 0)
    first = scheduler.select_destination(q, "first")
    assert first.cached_tokens == 0
    second = scheduler.select_destination(q, "second")
    assert (second.status, second.reason, second.cached_tokens) == (
        "wait",
        "pending_prefix",
        0,
    )
    target = a if first.replica == "cache" else b
    target["work"]["prefix_nodes"] = catalog("m/weight_v000007", [42] * 8192)
    scheduler.update_snapshot(json.dumps([a, b]))
    second = scheduler.select_destination(q, "second", 0.5)
    assert (second.replica, second.cached_tokens) == (first.replica, 8128)


def test_pending_prefix_expires_and_does_not_block_other_groups_or_versions():
    scheduler, q, _, _ = wait_router(0, 0, 0)
    scheduler.select_destination(q, "first")
    for adapter in ["m/weight_v000007", "m/weight_v000008"]:
        other = Request(adapter, "different-group", [42] * 8192, 1)
        assert scheduler.select_destination(other, adapter).status == "route"
    assert scheduler.select_destination(q, "second", 2.01).status == "route"


@pytest.mark.parametrize("outcome", ["busy", "failed", "cancelled", "ok"])
def test_completed_leader_cannot_leave_a_pending_prefix_wait(outcome):
    scheduler, q, _, _ = wait_router(0, 0, 0)
    scheduler.select_destination(q, "first")
    scheduler.complete("first", outcome)
    assert scheduler.select_destination(q, "second").status == "route"


def test_pending_prefix_spills_if_leader_becomes_overloaded():
    scheduler, q, a, b = wait_router(0, 0, 0)
    first = scheduler.select_destination(q, "first")
    target = a if first.replica == "cache" else b
    target["active"] = 20
    scheduler.update_snapshot(json.dumps([a, b]))
    second = scheduler.select_destination(q, "second")
    assert second.status == "route" and second.replica != first.replica


def test_short_prompt_has_no_pending_prefix_wait():
    scheduler, _, _, _ = wait_router(0, 0, 0)
    q = Request("m/weight_v000007", "group", [42] * 64, 1)
    scheduler.select_destination(q, "first")
    assert scheduler.select_destination(q, "second").status == "route"


def test_async_wait_honors_subpoll_prefill_budget():
    async def run():
        scheduler, q, a, b = wait_router(cached=1024)
        router = Router(
            AsyncMock(return_value=[a, b]),
            policy="cache_wait",
            capacity=32,
            rates={**RATES, "prefill_tokens_per_second": 100000.0},
            close=AsyncMock(),
        )
        router.scheduler.update_snapshot(json.dumps([a, b]))
        q = router.prepare_request("m/weight_v000007", "group", [42] * 1024, 1)
        loop = asyncio.get_running_loop()
        started = loop.time()
        selected = await router.select_destination(q, "r", deadline=started + 1)
        elapsed = loop.time() - started
        assert selected.replica == "other"
        assert 0.009 <= elapsed < 0.20  # The normal polling jitter is >=250 ms.
        router.complete("r")
        await router.aclose()

    asyncio.run(run())
