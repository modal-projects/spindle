import time
from unittest.mock import AsyncMock

import pytest

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_simple import SIMPLE_POLICIES, SimpleRouter
from spindle.inference.routing_work import WorkRates, token_prefixes


def pool(policy):
    router = SimpleRouter(
        AsyncMock(), policy=policy, rates=WorkRates(1000, 100, 0.1, 1, 8), slots=2
    )
    catalog = [f"m-{i}/v1" for i in range(8)]
    router.replicas = {
        h: Replica(
            h,
            work=dict(
                age_seconds=0,
                prefixes=frozenset(),
                prefill_tokens=0,
                decode_tokens=0,
                pending_adapters=[],
                request_ids=frozenset(),
                gpu_adapters=[],
                cpu_adapters=catalog,
            ),
        )
        for h in ("a", "b", "c", "d")
    }
    router.updated = time.monotonic()
    return router


def choose(router, rid, **kwargs):
    return router.select_destination(
        "m-0",
        "m-0/v1",
        rid,
        prompt=list(range(1024)),
        output_tokens=100,
        reservation_id=rid,
        **kwargs,
    )


@pytest.mark.parametrize("policy", SIMPLE_POLICIES)
def test_stale_state_falls_back_and_reservations_release(policy):
    router = pool(policy)
    host = choose(router, "one")
    router.release(host, "m-0/v1", reservation_id="one")
    assert not router.reservations and not router.host_reservations
    for r in router.replicas.values():
        r.work["age_seconds"] = 4
    assert choose(router, "stale") is None


def test_static_adapter_home_ignores_congestion_by_definition():
    router = pool("adapter-home")
    primary = choose(router, "first")
    router.replicas[primary].active = 1000
    assert {choose(router, str(i)) for i in range(20)} == {primary}


def test_two_choices_balance_only_within_fixed_adapter_pair():
    router = pool("adapter-two")
    choices = [choose(router, str(i)) for i in range(40)]
    assert len(set(choices)) == 2
    assert all(choices.count(h) == 20 for h in set(choices))


@pytest.mark.parametrize("policy", ["adapter-spill", "balanced-spill"])
def test_adapter_spill_preserves_home_until_it_is_overloaded(policy):
    router = pool(policy)
    home = choose(router, "first")
    router.replicas[home].active = 100
    assert choose(router, "overflow") != home


def test_balanced_home_equalizes_small_catalog_even_when_slots_are_plentiful():
    router = pool("balanced-home")
    router.slots = 32
    hosts = tuple(router.replicas)
    assignments = [router.primary_placement(f"m-{i}/v1", hosts)[0] for i in range(8)]
    assert {assignments.count(h) for h in hosts} == {2}


def test_least_load_counts_observed_local_request_only_once():
    router = pool("least-load")
    first = choose(router, "first")
    router.replicas[first].active = 1
    router.replicas[first].work["request_ids"] = frozenset({"first"})
    for h, r in router.replicas.items():
        if h != first:
            r.active = 2
    details = {}
    assert choose(router, "second", details=details) == first
    assert details["observed_and_reserved_requests"] == 1


def test_bounded_cache_preference_spills_when_home_is_overloaded():
    router = pool("bounded-kv")
    router.replicas["a"].work["prefixes"] = frozenset(
        h for h, _ in token_prefixes("m-0/v1", list(range(1024)))
    )
    router.replicas["a"].active = 1
    assert choose(router, "warm") == "a"
    router.replicas["a"].active = 100
    assert choose(router, "overflow") != "a"


@pytest.mark.parametrize("policy", SIMPLE_POLICIES)
def test_chunks_stay_together_then_release(policy):
    router = pool(policy)
    routes = [
        choose(router, str(i), group_id="g", group_size=16, sample_index=i)
        for i in range(16)
    ]
    assert len(set(routes[:8])) == len(set(routes[8:])) == 1
    for i, host in enumerate(routes):
        router.release(host, "m-0/v1", reservation_id=str(i))
    assert not router.reservations and not router.host_reservations
