import asyncio
from collections import Counter
import time
from unittest.mock import AsyncMock, patch

import httpx

from spindle.inference.lora_routing import LoraRouter, Replica, prefix_key
from spindle.inference.sampling import sample_task
from tests.inference.test_sampling import response, task


def router(replicas, **kwargs):
    result = LoraRouter(AsyncMock(return_value=replicas), **kwargs)
    result.replicas = {r.replica_id: r for r in replicas}
    result.updated = time.monotonic()
    result.next_refresh = result.updated + 100
    return result


def test_prefix_placement_stays_local_until_target_then_spills():
    pool = router(
        [Replica("a", recent=("model/weight_v000001",)), Replica("b")], target=4
    )
    routes = [
        pool.select_destination("model", "model/weight_v000001", "prefix")
        for i in range(12)
    ]
    assert len(set(routes[:4])) == 1
    assert routes[4] != routes[0]
    assert set(routes) == {"a", "b"}
    assert abs(routes.count("a") - routes.count("b")) <= 4
    for upstream in routes:
        pool.release(upstream, "model/weight_v000001")
    assert not any(pool.local.values())


def test_rendezvous_is_independent_of_worker_and_discovery_order():
    replicas = [Replica(str(i)) for i in range(8)]
    left, right = router(replicas), router(replicas[::-1])
    for i in range(50):
        assert left.select_destination(
            str(i), f"{i}/v1", "request"
        ) == right.select_destination(str(i), f"{i}/v1", "request")


def test_full_working_set_is_avoided_on_cold_placement():
    pool = router([Replica("a"), Replica("b")], slots=1)
    reference = router([Replica("a"), Replica("b")], slots=1)
    home = reference.select_destination("model", "new", "new-prefix")
    pool.replicas[home].adapters = {"other": 1}
    assert pool.select_destination("model", "new", "new-prefix") != home


def test_versions_and_models_do_not_share_warmth():
    pool = router([Replica("a"), Replica("b")])
    home = pool.select_destination("model", "model/v2", "request")
    pool.release(home, "model/v2")
    other = "a" if home == "b" else "b"
    pool.replicas[other].recent = ("another-model/v2", "model/v1")
    assert pool.select_destination("model", "model/v2", "request") == home


def test_prefix_home_survives_residency_changes_and_temporary_overflow():
    pool = router([Replica("a"), Replica("b")], target=4)
    home = pool.select_destination("m", "m/v1", "prefix")
    pool.release(home, "m/v1")
    other = next(h for h in pool.replicas if h != home)
    pool.replicas[other].recent = ("m/v1",)
    pool.replicas[other].prefixes = frozenset({prefix_key("m/v1", "prefix")})
    assert pool.select_destination("m", "m/v1", "prefix") == home
    pool.release(home, "m/v1")
    pool.replicas[home].active = 20
    assert pool.select_destination("m", "m/v1", "prefix") == other
    pool.release(other, "m/v1")
    pool.replicas[home].active = 0
    assert pool.select_destination("m", "m/v1", "prefix") == home


def test_another_frontend_recovers_version_scoped_prefix_home():
    pool = router([Replica("a"), Replica("b")])
    default = pool.select_destination("m", "m/v1", "prefix")
    other = next(h for h in pool.replicas if h != default)
    fresh = router(
        [
            Replica(default),
            Replica(other, prefixes=frozenset({prefix_key("m/v1", "prefix")})),
        ]
    )
    assert fresh.select_destination("m", "m/v1", "prefix") == other
    fresh.release(other, "m/v1")
    assert prefix_key("m/v1", "prefix") != prefix_key("m/v2", "prefix")
    reference = router([Replica(default), Replica(other)])
    assert fresh.select_destination(
        "m", "m/v2", "prefix"
    ) == reference.select_destination("m", "m/v2", "prefix")


def test_stale_or_failed_replicas_fall_back_and_refresh_removes_departed_nodes():
    async def run():
        pool = router([Replica("a")])
        chosen = pool.select_destination("m", "v1", "r")
        pool.release(chosen, "v1", failed=True)
        assert pool.select_destination("m", "v1", "r") is None
        pool.replicas = {"b": Replica("b")}
        pool.updated -= 20
        assert pool.select_destination("m", "v1", "r") is None
        pool.discover.return_value = [Replica("c")]
        await pool._refresh()
        assert pool.select_destination("m", "v1", "r") == "c"

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
        assert pool.select_destination("m", "v", "r") == "a"
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
        home = pool.select_destination("m", "m/v1", "probe")
        pool.release(home, "m/v1")
        # These reservations already exist in the reported active count.
        pool.local[home] = Counter({"m/v1": 50})
        await pool._refresh()
        routes = [pool.select_destination("m", "m/v1", "probe") for i in range(34)]
        assert routes[:16] == [home] * 16
        assert routes[16] != home
        assert abs(routes.count("a") - routes.count("b")) <= pool.target

    asyncio.run(run())


def test_burst_after_idle_waits_for_one_refresh_and_failed_discovery_falls_back():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()

        async def discover():
            started.set()
            await release.wait()
            return [Replica("warm", recent=("m/v1",))]

        pool = LoraRouter(discover)
        pool.replicas = {"old": Replica("old")}
        pool.updated = time.monotonic() - 20
        waiting = [asyncio.create_task(pool.refresh()) for _ in range(16)]
        await started.wait()
        assert all(not task.done() for task in waiting)
        pending = pool.pending
        release.set()
        await asyncio.gather(*waiting)
        assert pool.pending is pending
        assert pool.select_destination("m", "m/v1", "r") == "warm"
        pool.release("warm", "m/v1")

        pool.updated -= 20
        pool.next_refresh = 0
        pool.discover = AsyncMock(side_effect=TimeoutError)
        await pool.refresh()
        assert pool.select_destination("m", "m/v1", "r") is None

    asyncio.run(run())


def test_saturated_pool_preserves_small_prefix_groups_and_bounds_hot_overflow():
    pool = router([Replica(str(i), active=96) for i in range(8)], target=64)
    routes = [pool.select_destination("m", "m/v1", "problem") for _ in range(128)]
    # Saturation alone must not split an eight-sample training problem 8 ways.
    assert len(set(routes[:8])) == 1
    assert len(set(routes[:64])) == 1
    assert routes[64] != routes[0]
    assert max(Counter(routes).values()) == 64
    home = pool.prefix_homes[prefix_key("m/v1", "problem")]
    for route in routes:
        pool.release(route, "m/v1")
    assert pool.select_destination("m", "m/v1", "problem") == home


def test_new_prefixes_spread_when_adapter_capacity_is_available():
    pool = router([Replica(str(i)) for i in range(8)], target=64)
    destinations = [
        pool.select_destination("m", "m/v1", f"problem-{i}") for i in range(64)
    ]
    assert len(set(destinations)) == 8


def test_observations_refresh_during_long_generations_without_more_submissions():
    async def run():
        pool = LoraRouter(AsyncMock(return_value=[Replica("a")]), refresh_interval=0.01)
        await pool.refresh()
        chosen = pool.select_destination("m", "m/v1", "p")
        async with asyncio.timeout(0.5):
            while pool.discover.await_count < 3:
                await asyncio.sleep(0.01)
        pool.release(chosen, "m/v1")
        await pool.poller
        count = pool.discover.await_count
        await asyncio.sleep(0.03)
        assert pool.discover.await_count == count
        await pool.aclose()

    asyncio.run(run())


def test_two_choices_balance_a_new_prefix_without_moving_its_established_home():
    reference = router([Replica("a"), Replica("b")], target=64)
    first_hash_choice = reference.select_destination("m", "m/v1", "problem")
    pool = router([Replica("a"), Replica("b")], target=64)
    pool.replicas[first_hash_choice].active = 1
    home = pool.select_destination("m", "m/v1", "problem")
    assert home != first_hash_choice
    pool.release(home, "m/v1")
    pool.replicas[first_hash_choice].active = 0
    pool.replicas[home].active = 2
    assert pool.select_destination("m", "m/v1", "problem") == home
