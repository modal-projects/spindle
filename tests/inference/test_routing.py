import asyncio
from collections import Counter
import time
from unittest.mock import AsyncMock

import pytest

from spindle.inference.model_placement import ModelPlacement
from spindle.inference.routing import (
    Engine,
    InferenceRouter,
    PoolSnapshot,
    RoutingRequest,
)


def pool():
    router = InferenceRouter(AsyncMock(), ModelPlacement(capacity=8))
    catalog = tuple(f"model-{i}/weight_v000001" for i in range(8))
    router.engines = {
        host: Engine(host, host + ":8000", 0, frozenset(), catalog, (), 0)
        for host in ("a", "b", "c", "d")
    }
    router.updated = time.monotonic()
    return router


def choose(router, rid, **kwargs):
    details = kwargs.pop("details", None)
    engine = router.select_engine(
        RoutingRequest("model-0/weight_v000001", "conversation", rid, **kwargs),
        details=details,
    )
    return engine.replica_id if engine is not None else None


def snapshot(router):
    return PoolSnapshot(
        router.engines, {h: e.active for h, e in router.engines.items()}, router.updated
    )


def test_balanced_model_placement_survives_new_versions():
    router = pool()
    hosts = router.policy._placement_topology(
        "model-0/weight_v000001", snapshot(router)
    )
    primary = {
        model: router.policy._primary_placement(
            f"{model}/weight_v000001", hosts, snapshot(router)
        )[0]
        for model in [f"model-{i}" for i in range(8)]
    }
    assert set(Counter(primary.values()).values()) == {2}
    for replica in router.engines.values():
        replica.cpu_adapters += ("model-0/weight_v000002",)
    assert (
        router.policy._primary_placement(
            "model-0/weight_v000002", hosts, snapshot(router)
        )[0]
        == primary["model-0"]
    )


def test_busy_preferred_replica_spills_to_spare_capacity():
    router = pool()
    primary = choose(router, "one")
    router.finish_request("one")
    router.engines[primary].active = 100
    assert choose(router, "two") != primary


def test_group_chunks_preserve_siblings_but_new_chunks_can_spill():
    router = pool()
    first = choose(router, "0", group_id="group", group_size=16, sample_index=0)
    router.engines[first].active = 100
    assert choose(router, "1", group_id="group", group_size=16, sample_index=1) == first
    assert choose(router, "8", group_id="group", group_size=16, sample_index=8) != first


def test_observed_local_request_is_not_counted_twice():
    router = pool()
    first = choose(router, "one")
    router.engines[first].active = 1
    router.engines[first].request_ids = frozenset({"one"})
    details = {}
    assert choose(router, "two", details=details) == first
    assert details["observed_and_reserved_requests"] == 1
    router.finish_request("one")
    router.finish_request("two")
    assert not router.reservations and not router.host_reservations


def test_stale_observations_and_failed_hosts_fall_back():
    router = pool()
    for replica in router.engines.values():
        replica.age_seconds = 4
    assert choose(router, "stale") is None
    assert not router.reservations
    for replica in router.engines.values():
        replica.age_seconds = 0
    first = choose(router, "one")
    router.finish_request("one", failed=True)
    assert choose(router, "two") != first


def test_transient_host_failure_does_not_reassign_other_models():
    router = pool()
    hosts = router.policy._placement_topology(
        "model-0/weight_v000001", snapshot(router)
    )
    router.policy._primary_placement("model-0/weight_v000001", hosts, snapshot(router))
    old = dict(router.policy.primary_hosts)
    router.failed_until["a"] = time.monotonic() + 10
    choose(router, "one")
    assert router.policy.primary_hosts == old


def test_refresh_coalesces_waiters_and_cancel_does_not_cancel_discovery():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        observed = list(pool().engines.values())

        async def discover():
            entered.set()
            await release.wait()
            return observed

        discover = AsyncMock(side_effect=discover)
        router = InferenceRouter(discover, ModelPlacement(capacity=8))
        callers = [asyncio.create_task(router.refresh()) for _ in range(40)]
        try:
            await entered.wait()
            callers[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await callers[0]
            assert not router.pending.cancelled()
            release.set()
            await asyncio.gather(*callers[1:])
            assert discover.await_count == 1
            assert len(router.engines) == 4
        finally:
            release.set()
            await router.aclose()

    asyncio.run(scenario())


def test_fresh_snapshot_selection_does_not_wait_for_slow_poll():
    async def scenario():
        router = pool()
        release = asyncio.Event()
        router.discover = release.wait
        try:
            await asyncio.wait_for(router.refresh(), 0.1)
            assert not router.pending.done()
            assert choose(router, "one") is not None
        finally:
            await router.aclose()

    asyncio.run(scenario())


def test_policy_receives_eligible_loads_and_full_catalog():
    class LeastLoaded:
        def select_engine(self, request, pool):
            assert request.adapter == "model-0/weight_v000001"
            assert set(pool.engines) == {"a", "b", "c", "d"}
            assert set(pool.loads) == {"a", "b"}
            # One local reservation is already visible in the engine snapshot.
            assert pool.loads == {"a": 2, "b": 1}
            return min(pool.loads, key=pool.loads.get)

    router = pool()
    router.policy = LeastLoaded()
    router.engines["a"].active = 1
    router.engines["a"].request_ids = frozenset({"observed"})
    router.host_reservations["a"] = {"observed", "unobserved"}
    router.engines["b"].active = 1
    router.engines["c"].age_seconds = 4
    router.failed_until["d"] = time.monotonic() + 5
    assert choose(router, "request") == "b"
    router.finish_request("request", failed=True)
    assert router.failed_until["b"] > time.monotonic()
    assert "b" not in router.host_reservations


@pytest.mark.parametrize("destination", [None, "unknown"])
def test_policy_can_defer_but_cannot_select_an_unavailable_engine(destination):
    class Policy:
        def select_engine(self, request, pool):
            return destination

    router = pool()
    router.policy = Policy()
    if destination is None:
        assert choose(router, "request") is None
    else:
        with pytest.raises(ValueError, match="ineligible"):
            choose(router, "request")
    assert not router.reservations


def test_failed_discovery_does_not_renew_stale_observations():
    async def scenario():
        router = pool()
        router.updated -= 11
        previous = router.updated
        router.discover.side_effect = RuntimeError("discovery unavailable")
        try:
            await router.refresh()
            assert router.updated == previous
            assert choose(router, "request") is None
        finally:
            await router.aclose()

    asyncio.run(scenario())
