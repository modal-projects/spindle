import time
from collections import Counter
from unittest.mock import AsyncMock

from spindle.inference.lora_routing import Replica, _ranking
from spindle.inference.routing_domains import DomainRouter
from spindle.inference.routing_work import WorkRates, token_prefixes


def pool():
    router = DomainRouter(AsyncMock(), rates=WorkRates(1000, 100, 0.1, 1, 8))
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
                cpu_adapters=[],
            ),
        )
        for h in ("a", "b", "c", "d")
    }
    router.updated = time.monotonic()
    return router


def choose(router, rid, **kwargs):
    return router.select_destination(
        "model",
        "model/v1",
        rid,
        prompt=list(range(1024)),
        output_tokens=100,
        reservation_id=rid,
        **kwargs,
    )


def test_cold_small_adapter_concentrates_then_expands_with_demand():
    router = pool()
    primary = _ranking("model", tuple(sorted(router.replicas)))[0]
    routes = [choose(router, str(i)) for i in range(16)]
    assert routes[:8] == [primary] * 8
    assert len(set(routes)) == 2
    for i, host in enumerate(routes):
        router.release(host, "model/v1", reservation_id=str(i))
    assert not router.reservations and not router.host_reservations


def test_domain_congestion_escapes_even_for_cold_adapter():
    router = pool()
    primary = _ranking("model", tuple(sorted(router.replicas)))[0]
    router.replicas[primary].work["decode_tokens"] = 100000
    assert choose(router, "request") != primary


def test_real_cache_hit_outside_domain_is_preserved():
    router = pool()
    outside = _ranking("model", tuple(sorted(router.replicas)))[-1]
    router.replicas[outside].work["prefixes"] = frozenset(
        h for h, n in token_prefixes("model/v1", list(range(1024)))
    )
    assert choose(router, "request") == outside


def test_observed_local_demand_is_not_counted_twice():
    router = pool()
    primary = choose(router, "first")
    router.replicas[primary].adapters = {"model/v1": 1}
    router.replicas[primary].work["request_ids"] = frozenset({"first"})
    detail = {}
    choose(router, "second", details=detail)
    assert detail["adapter_demand"] == 2


def test_balanced_primary_catalog_respects_gpu_slot_budget():
    router = pool()
    router.balanced = True
    router.slots = 4
    catalog = [f"model-{i}/v1" for i in range(16)]
    for replica in router.replicas.values():
        replica.work["cpu_adapters"] = list(reversed(catalog))
    hosts = tuple(sorted(router.replicas))
    counts = Counter(router.primary_placement(a, hosts)[0] for a in catalog)
    assert set(counts.values()) == {4}
    before = dict(router.primary_hosts)
    for replica in router.replicas.values():
        replica.work["cpu_adapters"] = catalog
    for adapter in reversed(catalog):
        router.primary_placement(adapter, hosts)
    assert router.primary_hosts == before


def test_one_slow_observation_does_not_remap_every_primary():
    router = pool()
    router.balanced = router.stable = True
    router.slots = 4
    catalog = [f"model-{i}/v1" for i in range(16)]
    for replica in router.replicas.values():
        replica.work["cpu_adapters"] = catalog
    hosts = router.placement_topology(catalog[0])
    router.primary_placement(catalog[0], hosts)
    before = dict(router.primary_hosts)
    del router.replicas["a"]
    router.updated = time.monotonic()
    hosts = router.placement_topology(catalog[0])
    assert "a" in hosts
    router.primary_placement(catalog[0], hosts)
    assert router.primary_hosts == before
    router.host_seen["a"] -= router.stale_after + 1
    assert "a" not in router.placement_topology(catalog[0])
