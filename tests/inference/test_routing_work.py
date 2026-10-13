import asyncio
from array import array
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_work import WorkRates, WorkRouter, token_prefixes
from spindle.inference import sglang_routing_observer as observer


def state(**changes):
    return (
        dict(
            age_seconds=0,
            prefixes=frozenset(),
            prefill_tokens=0,
            decode_tokens=0,
            pending_adapters=[],
            request_ids=frozenset(),
            gpu_adapters=[],
            cpu_adapters=[],
        )
        | changes
    )


def router(**states):
    replicas = [Replica(h, work=s) for h, s in states.items()]
    result = WorkRouter(
        AsyncMock(return_value=replicas),
        rates=WorkRates(1000, 100, 0.1, 2),
    )
    result.replicas = {r.replica_id: r for r in replicas}
    result.updated = time.monotonic()
    return result


def choose(pool, rid="r", adapter="m/v1", **kwargs):
    return pool.select_destination(
        "m",
        adapter,
        "affinity",
        prompt=list(range(1024)),
        output_tokens=100,
        reservation_id=rid,
        **kwargs,
    )


def test_kv_saving_can_outweigh_gpu_adapter_warmth():
    cached = frozenset(h for h, n in token_prefixes("m/v1", list(range(1024))))
    pool = router(
        a=state(prefixes=cached, cpu_adapters=["m/v1"]),
        b=state(gpu_adapters=["m/v1"], cpu_adapters=["m/v1"]),
    )
    assert choose(pool) == "a"
    pool.release("a", "m/v1", reservation_id="r")
    pool.replicas["a"].work["decode_tokens"] = 1000
    assert choose(pool) == "b"


def test_version_change_gets_no_old_version_cache_credit():
    cached = frozenset(h for h, n in token_prefixes("m/v1", list(range(1024))))
    pool = router(
        a=state(prefixes=cached), b=state(cpu_adapters=["m/v2"], gpu_adapters=["m/v2"])
    )
    assert choose(pool, adapter="m/v2") == "b"


def test_synchronous_registration_is_charged_once_as_scheduler_work():
    pool = router(a=state(decode_tokens=100))
    details = {}
    choose(pool, details=details)
    assert details["backlog_seconds"] == 1
    assert details["registration_seconds"] == 2
    assert details["score"] == 1 + 2 + 0.1 + 1.024 + 1
    next_details = {}
    choose(pool, "next", details=next_details)
    assert next_details["registration_seconds"] == 0
    assert next_details["transfer_seconds"] == 0
    assert abs(next_details["backlog_seconds"] - (1 + 2.024 + 2 + 0.1)) < 1e-9


def test_burst_reservations_spill_and_observed_requests_are_not_counted_twice():
    pool = router(a=state(), b=state())
    routes = [choose(pool, str(i)) for i in range(20)]
    assert abs(routes.count("a") - routes.count("b")) <= 1
    chosen = routes[0]
    existing = pool.host_reservations[chosen]
    pool.replicas[chosen].work.update(
        request_ids=frozenset(existing),
        prefill_tokens=len(existing) * 1024,
        decode_tokens=len(existing) * 100,
        pending_adapters=["m/v1"],
    )
    details = {}
    pool.replicas = {chosen: pool.replicas[chosen]}
    choose(pool, "next", details=details)
    assert (
        abs(details["backlog_seconds"] - (len(existing) - 1) * 2.024 - 0.1 - 2) < 1e-9
    )
    for i, host in enumerate(routes):
        pool.release(host, "m/v1", reservation_id=str(i))
    pool.release(chosen, "m/v1", reservation_id="next")
    assert not pool.reservations and not any(pool.local.values())


def test_old_engine_snapshot_falls_back_even_when_http_poll_is_fresh():
    pool = router(a=state(age_seconds=4))
    assert choose(pool) is None


def test_siblings_stay_together_and_large_groups_split_at_engine_capacity():
    pool = router(a=state(), b=state())
    pool.rates = WorkRates(1000, 100, 0.1, 2, decode_concurrency=4)
    routes = [
        choose(pool, str(i), group_id="rollout", group_size=12, sample_index=i)
        for i in range(12)
    ]
    assert len(set(routes[:4])) == 1
    assert len(set(routes[4:8])) == 1
    assert routes[0] != routes[4]
    assert len(set(routes[8:])) == 1
    for i, host in enumerate(routes):
        pool.release(host, "m/v1", reservation_id=str(i))
    assert not any(pool.local.values())


def test_group_cost_preserves_kv_when_individual_greedy_placement_would_spill():
    cached = frozenset(h for h, n in token_prefixes("m/v1", list(range(1024))))
    pool = router(
        a=state(
            prefixes=cached,
            decode_tokens=300,
            cpu_adapters=["m/v1"],
            gpu_adapters=["m/v1"],
        ),
        b=state(cpu_adapters=["m/v1"], gpu_adapters=["m/v1"]),
    )
    pool.rates = WorkRates(1000, 100, 0.1, 2, decode_concurrency=8)
    assert choose(pool, "single") == "b"
    pool.release("b", "m/v1", reservation_id="single")
    assert choose(pool, "group", group_id="g", group_size=8) == "a"


def test_prefix_preparation_never_uses_affinity_lookup_and_is_group_specific():
    async def run():
        pool = router(a=state())
        pool.resolve_affinity = AsyncMock()
        left = await pool.prepare_prefix("m/v1", "same", prompt=list(range(1024)))
        right = await pool.prepare_prefix("m/v1", "same", prompt=[2] * 1024)
        assert left != right
        pool.resolve_affinity.assert_not_called()

    asyncio.run(run())


class Key:
    def __init__(self, tokens, uid="uid"):
        self.tokens, self.extra_key, self.is_bigram = array("q", tokens), uid, False

    def raw_token_ids(self):
        return self.tokens

    def __len__(self):
        return len(self.tokens)


class MambaRadixCache:
    pass


def test_observer_requires_live_kv_path_and_mamba_boundary_and_removes_evictions():
    observer.hashes.clear()
    first = SimpleNamespace(
        id=1,
        key=Key(range(64)),
        children={},
        value=[1],
        mamba_value=None,
        last_access_time=1,
    )
    second = SimpleNamespace(
        id=2,
        key=Key(range(64, 128)),
        children={},
        value=[1],
        mamba_value=[1],
        last_access_time=2,
    )
    first.children[2] = second
    cache = MambaRadixCache()
    cache.root_node = SimpleNamespace(children={1: first})
    expected = token_prefixes("m/v1", list(range(129)))
    assert observer.device_prefixes(cache, {"uid": "m/v1"}) == [expected[1][0]]
    second.mamba_value = None
    assert observer.device_prefixes(cache, {"uid": "m/v1"}) == []
    second.mamba_value = [1]
    first.value = None
    assert observer.device_prefixes(cache, {"uid": "m/v1"}) == []
    assert not observer.hashes


def test_salted_or_unknown_adapter_ids_never_receive_credit():
    observer.hashes.clear()
    cache = MambaRadixCache()
    cache.root_node = SimpleNamespace(
        children={
            1: SimpleNamespace(
                id=1,
                key=Key(range(64), "salt-uid"),
                children={},
                value=[1],
                mamba_value=[1],
                last_access_time=1,
            )
        }
    )
    assert observer.device_prefixes(cache, {"uid": "m/v1"}) == []
