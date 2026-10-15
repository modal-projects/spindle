import time
from unittest.mock import AsyncMock

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_dlora import DloraRouter
from spindle.inference.routing_work import WorkRates, token_prefixes
from spindle.inference.sglang_routing_observer import install


def router(**states):
    result = DloraRouter(AsyncMock(), rates=WorkRates(1000, 100, 0.1, 1, 4))
    result.updated = time.monotonic()
    for host, extra in states.items():
        state = dict(
            age_seconds=0,
            prefixes=frozenset(),
            prefill_tokens=0,
            decode_tokens=0,
            pending_adapters=[],
            request_ids=frozenset(),
            cpu_adapters=["a/v1"],
            gpu_adapters=["a/v1"],
        )
        result.replicas[host] = Replica(host, work=state | extra)
    return result


def choose(pool, ident="r", adapter="a/v1"):
    return pool.select_destination(
        "a",
        adapter,
        "conversation",
        prompt=list(range(8192)),
        output_tokens=100,
        reservation_id=ident,
    )


def test_wait_for_warm_prefix_until_queue_delay_exceeds_rebuild():
    cached = frozenset(h for h, _ in token_prefixes("a/v1", list(range(8192))))
    pool = router(warm=dict(prefixes=cached, decode_tokens=100), cold={})
    assert choose(pool) == "warm"
    pool.release("warm", "a/v1", reservation_id="r")
    pool.replicas["warm"].work["decode_tokens"] = 2000
    assert choose(pool) == "cold"


def test_busy_merged_replica_does_not_accept_another_version_when_mixed_worker_exists():
    pool = router(
        merged=dict(
            decode_tokens=400,
            adapter_decode_maxima={"a/v1": 100},
            mode=dict(
                merged_adapter="a/v1",
                running=4,
                profile=dict(
                    merged_token_seconds=0.03,
                    unmerge_seconds=0.5,
                ),
            ),
        ),
        mixed=dict(decode_tokens=1),
    )
    assert choose(pool, adapter="a/v2") == "mixed"
    del pool.replicas["mixed"]
    # New versions remain serviceable even when all engines are specialized.
    assert choose(pool, ident="new", adapter="a/v2") == "merged"


def test_overloaded_mixed_engine_does_not_strand_demand_behind_specialization():
    pool = router(
        specialized=dict(
            decode_tokens=400,
            adapter_decode_maxima={"a/v1": 100},
            mode=dict(
                merged_adapter="a/v1",
                running=4,
                profile=dict(
                    merged_token_seconds=0.03,
                    unmerge_seconds=0.5,
                ),
            ),
        ),
        mixed=dict(decode_tokens=2000),
    )
    # Three seconds to drain + restoration beats twenty seconds of mixed work.
    assert choose(pool, adapter="a/v2") == "specialized"


def test_failed_and_transitioning_engines_are_not_selected_as_fallbacks():
    pool = router(
        failed=dict(mode=dict(failed="uncertain merge")),
        transitioning=dict(mode=dict(transitioning=True)),
    )
    assert choose(pool) is None
    assert not pool.host_reservations


def test_partial_merged_batch_pays_drain_latency_before_switching():
    pool = router(
        specialized=dict(
            decode_tokens=100,
            adapter_decode_maxima={"a/v1": 100},
            mode=dict(
                merged_adapter="a/v1",
                running=1,
                profile=dict(merged_token_seconds=0.1, unmerge_seconds=0.5),
            ),
        ),
        mixed=dict(decode_tokens=500),
    )
    # Aggregate full-batch throughput says one second; this request actually
    # blocks restoration for ten seconds. Five seconds of mixed work is better.
    assert choose(pool, adapter="a/v2") == "mixed"


def test_local_reservations_expand_hot_adapter_without_permanent_home():
    pool = router(a={}, b={}, c={})
    routes = [choose(pool, str(i)) for i in range(24)]
    assert len(set(routes)) == 3
    for i, host in enumerate(routes):
        pool.release(host, "a/v1", reservation_id=str(i))
    assert not pool.host_reservations


def test_pr_observer_uses_current_request_prefix_field(tmp_path):
    scheduler = tmp_path / "srt/managers/scheduler.py"
    scheduler.parent.mkdir(parents=True)
    scheduler.write_text(
        "import logging\nclass Scheduler:\n    def step(self):\n        self.process_pending_chunked_abort()\n"
    )
    install(tmp_path, modern=True)
    source = (tmp_path / "srt/spindle_routing_observer.py").read_text()
    assert "else max(0, len(req.origin_input_ids) - req.prefix_len)" in source
    assert "or get_parallel().dp_size != 1" in source
    compile(source, str(tmp_path), "exec")
