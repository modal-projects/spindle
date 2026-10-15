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
        merged=dict(mode=dict(merged_adapter="a/v1", running=4)),
        mixed=dict(decode_tokens=1),
    )
    assert choose(pool, adapter="a/v2") == "mixed"
    del pool.replicas["mixed"]
    # New versions remain serviceable even when all engines are specialized.
    assert choose(pool, ident="new", adapter="a/v2") == "merged"


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
