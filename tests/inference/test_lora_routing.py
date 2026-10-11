import asyncio
from collections import Counter
import time
from unittest.mock import AsyncMock, patch

import httpx

from spindle.inference.lora_routing import LoraRouter, Replica
from spindle.inference.sampling import sample_task
from tests.inference.test_sampling import response, task


def router(replicas, **kwargs):
    result = LoraRouter(AsyncMock(return_value=replicas), **kwargs)
    result.replicas = {r.replica_id: r for r in replicas}
    result.updated = time.monotonic()
    result.next_refresh = result.updated + 100
    return result


def test_hot_adapter_stays_local_and_hot_load_spills_without_losing_reservations():
    pool = router(
        [Replica("a", recent=("model/weight_v000001",)), Replica("b")], target=4
    )
    routes = [pool.acquire("model", "model/weight_v000001", str(i)) for i in range(12)]
    assert routes[:2] == ["a"] * 2
    assert routes[2] == "b"
    assert set(routes) == {"a", "b"}
    assert abs(routes.count("a") - routes.count("b")) <= 4
    for upstream in routes:
        pool.release(upstream, "model/weight_v000001")
    assert not any(pool.local.values())


def test_rendezvous_is_independent_of_worker_and_discovery_order():
    replicas = [Replica(str(i)) for i in range(8)]
    left, right = router(replicas), router(replicas[::-1])
    for i in range(50):
        assert left.acquire(str(i), f"{i}/v1", "request") == right.acquire(
            str(i), f"{i}/v1", "request"
        )


def test_full_working_set_is_avoided_on_cold_placement():
    pool = router([Replica("a"), Replica("b")], slots=1)
    home = pool.acquire("model", "new", "request")
    pool.release(home, "new")
    pool.replicas[home].adapters = {"other": 1}
    assert pool.acquire("model", "new", "request") != home


def test_versions_and_models_do_not_share_warmth():
    pool = router([Replica("a"), Replica("b")])
    home = pool.acquire("model", "model/v2", "request")
    pool.release(home, "model/v2")
    other = "a" if home == "b" else "b"
    pool.replicas[other].recent = ("another-model/v2", "model/v1")
    assert pool.acquire("model", "model/v2", "request") == home


def test_stale_or_failed_replicas_fall_back_and_refresh_removes_departed_nodes():
    async def run():
        pool = router([Replica("a")])
        chosen = pool.acquire("m", "v1", "r")
        pool.release(chosen, "v1", failed=True)
        assert pool.acquire("m", "v1", "r") is None
        pool.replicas = {"b": Replica("b")}
        pool.updated -= 20
        assert pool.acquire("m", "v1", "r") is None
        pool.discover.return_value = [Replica("c")]
        await pool._refresh()
        assert pool.acquire("m", "v1", "r") == "c"

    asyncio.run(run())


def test_refresh_is_coalesced_and_does_not_block_warm_requests():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def discover():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return [Replica("a")]

        pool = LoraRouter(discover)
        initial = [asyncio.create_task(pool.refresh()) for _ in range(20)]
        await started.wait()
        assert calls == 1
        initial[0].cancel()
        release.set()
        await asyncio.gather(*initial, return_exceptions=True)
        pool.next_refresh = 0
        release.clear()
        await asyncio.wait_for(pool.refresh(), 0.1)
        assert pool.acquire("m", "v", "r") == "a"
        release.set()
        await pool.pending

    asyncio.run(run())


def test_flash_remap_and_version_mismatch_keep_constraints_and_release_load():
    async def run():
        pool = router(
            [Replica("a", upstream="route-a"), Replica("b", upstream="route-b")]
        )
        requests = []

        def handle(request):
            requests.append(request)
            result = response(6 if len(requests) == 1 else 7)
            result.headers["x-spindle-replica"] = "remapped"
            return result

        with patch("spindle.inference.sampling._backoff", new=AsyncMock()):
            result = await sample_task(
                task(),
                "https://gateway",
                transport=httpx.MockTransport(handle),
                lora_router=pool,
            )
        assert len(requests) == 2
        assert requests[0].headers["Modal-Flash-Upstream"].startswith("route-")
        assert "Modal-Session-ID" not in requests[0].headers
        assert result["sequences"][0]["tokens"] == [5]
        assert not any(pool.local.values())

    asyncio.run(run())


def test_cancelled_generation_releases_router_reservation():
    async def run():
        started = asyncio.Event()
        pool = router([Replica("a", upstream="route-a")])

        async def handle(request):
            started.set()
            await asyncio.Future()

        pending = asyncio.create_task(
            sample_task(
                task(),
                "https://gateway",
                transport=httpx.MockTransport(handle),
                lora_router=pool,
            )
        )
        await started.wait()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        assert not any(pool.local.values())

    asyncio.run(run())


def test_exact_selection_overrides_shared_client_affinity_headers():
    async def run():
        pool = router([Replica("a", upstream="host-a:443")])

        def handle(request):
            assert request.headers["Modal-Flash-Upstream"] == "host-a:443"
            assert "Modal-Session-ID" not in request.headers
            assert "Modal-Routing-Affinity-Key" not in request.headers
            return response(7)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            headers={"Modal-Session-ID": "legacy", "Modal-Routing-Affinity-Key": "new"},
        ) as client:
            await sample_task(
                task(), "https://gateway", client=client, lora_router=pool
            )
            assert client.headers["Modal-Session-ID"] == "legacy"

    asyncio.run(run())


def test_new_reservations_add_to_observed_load_without_double_counting_old_ones():
    async def run():
        pool = router(
            [Replica("a", active=100, recent=("m/v1",)), Replica("b", active=100)],
            target=16,
        )
        # These requests existed before the replica observation and are already
        # represented in its active count.
        pool.local["a"] = Counter({"m/v1": 50})
        await pool._refresh()
        routes = [pool.acquire("m", "m/v1", str(i)) for i in range(34)]
        assert routes[:33] == ["a"] * 33
        assert routes[33] == "b"

    asyncio.run(run())
