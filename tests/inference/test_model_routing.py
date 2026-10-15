import asyncio
from collections import Counter
import time
from unittest.mock import AsyncMock

import pytest

from spindle.inference.model_routing import ModelRouter, Replica


def pool():
    router = ModelRouter(AsyncMock(), capacity=8)
    catalog = tuple(f"model-{i}/weight_v000001" for i in range(8))
    router.replicas = {
        host: Replica(host, host + ":8000", 0, frozenset(), catalog, (), 0)
        for host in ("a", "b", "c", "d")
    }
    router.updated = time.monotonic()
    return router


def choose(router, rid, **kwargs):
    return router.select_destination(
        "model-0/weight_v000001", "conversation", reservation_id=rid, **kwargs
    )


def test_balanced_model_placement_survives_new_versions():
    router = pool()
    hosts = router._placement_topology("model-0/weight_v000001")
    primary = {
        model: router._primary_placement(f"{model}/weight_v000001", hosts)[0]
        for model in [f"model-{i}" for i in range(8)]
    }
    assert set(Counter(primary.values()).values()) == {2}
    for replica in router.replicas.values():
        replica.cpu_adapters += ("model-0/weight_v000002",)
    assert (
        router._primary_placement("model-0/weight_v000002", hosts)[0]
        == primary["model-0"]
    )


def test_busy_preferred_replica_spills_to_spare_capacity():
    router = pool()
    primary = choose(router, "one")
    router.release(primary, reservation_id="one")
    router.replicas[primary].active = 100
    assert choose(router, "two") != primary


def test_group_chunks_preserve_siblings_but_new_chunks_can_spill():
    router = pool()
    first = choose(router, "0", group_id="group", group_size=16, sample_index=0)
    router.replicas[first].active = 100
    assert choose(router, "1", group_id="group", group_size=16, sample_index=1) == first
    assert choose(router, "8", group_id="group", group_size=16, sample_index=8) != first


def test_observed_local_request_is_not_counted_twice():
    router = pool()
    first = choose(router, "one")
    router.replicas[first].active = 1
    router.replicas[first].request_ids = frozenset({"one"})
    details = {}
    assert choose(router, "two", details=details) == first
    assert details["observed_and_reserved_requests"] == 1
    router.release(first, reservation_id="one")
    router.release(first, reservation_id="two")
    assert not router.reservations and not router.host_reservations


def test_stale_observations_and_failed_hosts_fall_back():
    router = pool()
    for replica in router.replicas.values():
        replica.age_seconds = 4
    assert choose(router, "stale") is None
    assert not router.reservations
    for replica in router.replicas.values():
        replica.age_seconds = 0
    first = choose(router, "one")
    router.release(first, reservation_id="one", failed=True)
    assert choose(router, "two") != first


def test_transient_host_failure_does_not_reassign_other_models():
    router = pool()
    hosts = router._placement_topology("model-0/weight_v000001")
    router._primary_placement("model-0/weight_v000001", hosts)
    old = dict(router.primary_hosts)
    router.failed_until["a"] = time.monotonic() + 10
    choose(router, "one")
    assert router.primary_hosts == old


def test_refresh_coalesces_waiters_and_cancel_does_not_cancel_discovery():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        observed = list(pool().replicas.values())

        async def discover():
            entered.set()
            await release.wait()
            return observed

        discover = AsyncMock(side_effect=discover)
        router = ModelRouter(discover, capacity=8)
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
            assert len(router.replicas) == 4
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
