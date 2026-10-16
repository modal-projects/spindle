import time
from unittest.mock import AsyncMock

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_capacity import CapacityRouter, decode_iterations
from spindle.inference.routing_work import WorkRates, WorkRouter, token_prefixes


def pool(kind=CapacityRouter):
    router = kind(
        AsyncMock(), rates=WorkRates(1000, 100, 0.1, 2, 8), slots=2, adapter_costs=False
    )
    router.replicas = {}
    for host, tokens, adapters in (
        ("a", 800, {"m/v1": 100, "x/v1": 100}),
        ("b", 600, {f"other-{i}/v1": 100 for i in range(4)}),
    ):
        router.replicas[host] = Replica(
            host,
            work=dict(
                age_seconds=0,
                prefixes=frozenset(),
                prefill_tokens=0,
                decode_tokens=tokens,
                adapter_decode_maxima=adapters,
                pending_adapters=list(adapters),
                request_ids=frozenset(),
                gpu_adapters=list(adapters),
                cpu_adapters=list(adapters),
            ),
        )
    router.updated = time.monotonic()
    return router


def choose(router, rid="r", prompt=None, **kwargs):
    return router.select_destination(
        "m",
        "m/v1",
        "key",
        prompt=list(range(1024)) if prompt is None else prompt,
        output_tokens=100,
        reservation_id=rid,
        **kwargs,
    )


def test_iteration_bounds_identify_different_capacity_constraints():
    assert decode_iterations(100, {"a": 100}, 8, 2) == 100
    assert decode_iterations(1600, {"a": 100}, 8, 2) == 200
    assert decode_iterations(800, {str(i): 100 for i in range(8)}, 8, 2) == 400


def test_adapter_fragmentation_can_outweigh_lower_total_decode_work():
    assert choose(pool(WorkRouter)) == "b"
    assert choose(pool()) == "a"


def test_large_actual_prefix_saving_can_outweigh_adapter_fragmentation():
    router = pool()
    prompt = list(range(32768))
    router.replicas["b"].work["prefixes"] = frozenset(
        h for h, _ in token_prefixes("m/v1", prompt)
    )
    assert choose(router, prompt=prompt) == "b"


def test_observed_reservation_is_not_counted_twice_in_either_decode_bound():
    router = pool()
    router.replicas.pop("b")
    choose(router)
    router.replicas["a"].work.update(decode_tokens=900, request_ids=frozenset({"r"}))
    details = {}
    choose(router, "next", details=details)
    assert details["linear_decode_seconds"] == 10
    assert details["decode_iteration_bound"] == 125


def test_old_observer_schema_falls_back_without_reserving():
    router = pool()
    del router.replicas["a"].work["adapter_decode_maxima"]
    assert choose(router) is None
    assert not router.reservations
