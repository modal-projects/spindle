from pathlib import Path
import json
import asyncio
from copy import deepcopy
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx


BENCHMARK = Path(__file__).resolve().parents[2] / "scripts/benchmarks/lora_routing"


def client(monkeypatch, rewards):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    cls = runpy.run_path(str(BENCHMARK / "client.py"))["Client"]
    instance = cls.__new__(cls)
    instance.cfg = dict(
        groups_per_update=2,
        minimum_refill_groups=2,
        max_candidate_groups=4,
        max_tokens=8,
    )
    instance.next_group = 0
    instance.event = lambda *args, **kwargs: None
    instance.group = lambda index, snapshot: dict(
        prompt=[1, 2],
        sequences=[
            dict(tokens=[3, 4], logprobs=[-0.2, -0.3], reward=r) for r in rewards[index]
        ],
    )
    return instance


def test_refills_keep_exact_training_batch_and_account_for_unused_mixed_groups(
    monkeypatch,
):
    runner = client(monkeypatch, [[0, 1], [0, 0], [0, 1], [0, 1]])
    data, stats = runner.rollout((None, 0), 1)
    assert len(data) == 4
    assert stats["groups_sampled"] == 4
    assert stats["groups_filtered"] == stats["groups_unused"] == 1
    assert stats["groups_retained"] == 2
    assert stats["output_tokens"] == 16
    assert stats["used_output_tokens"] == 8


def test_refills_stop_at_candidate_budget_without_training_constant_reward_groups(
    monkeypatch,
):
    runner = client(monkeypatch, [[0, 0]] * 4)
    with pytest.raises(RuntimeError, match="Too few mixed-reward"):
        runner.rollout((None, 0), 1)
    assert runner.next_group == 4


def test_state_preparation_preserves_saved_states_and_keeps_trainer_nonempty(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    scope = runpy.run_path(str(BENCHMARK / "prepare_states.py"))
    active = set()
    first_creation = True

    class PreparedClient:
        def __init__(self, cfg, root, base_url, index):
            nonlocal first_creation
            if not first_creation:
                assert active, "The trainer must not become empty between preparations"
            first_creation = False
            self.index = index
            self.initial_state_path = f"saved-{index}"
            self.trainer = type("Trainer", (), {"model_id": f"model-{index}"})()
            active.add(index)

        def unload(self):
            active.remove(self.index)
            return True

    scope["main"].__globals__["Client"] = PreparedClient
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                model="Qwen/Qwen3.8-27B",
                updates=4,
                max_tokens=8192,
                context_tokens=16384,
            )
        )
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(["saved-0"]))
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_states.py",
            "--config",
            str(config),
            "--root",
            str(output),
            "--base-url",
            "http://test",
            "--clients",
            "4",
            "--resume-from",
            str(previous),
        ],
    )
    scope["main"]()
    assert active == {1}
    assert json.loads((output / "initialization-model.json").read_text()) == "model-1"
    measured = json.loads((output / "measured-config.json").read_text())
    assert measured["initial_state_paths"] == [f"saved-{i}" for i in range(4)]
    assert measured["save_initial_state"] is False


def test_benchmark_reschedule_fails_before_reset_or_generation(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    monkeypatch.setenv("MODAL_TASK_ID", "replacement-worker")
    scope = runpy.run_path(str(BENCHMARK / "cache_replay.py"))
    executions = SimpleNamespace(put=SimpleNamespace(aio=AsyncMock(return_value=False)))
    for function, arguments in (
        (scope["run_arm"], ("pool", "kv", [], {}, executions)),
        (scope["run_training_frontend"], ("pool", "kv", [], {}, None, 0, executions)),
    ):
        # Empty workload/config and no startup queue would fail if execution
        # reached setup. A rescheduled input must fail at its persistent claim.
        with pytest.raises(RuntimeError, match="Benchmark worker was rescheduled"):
            asyncio.run(function.local(*arguments))


def test_live_trainer_exit_invalidates_only_unfinished_arm(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    scope = runpy.run_path(str(BENCHMARK / "supervise.py"))
    check = scope["interrupted_trainer"]
    known = {
        "old": dict(task_id="old", gpu_count=8, finished_at=90),
        "inference": dict(task_id="inference", gpu_count=1, finished_at=110),
        "trainer": dict(task_id="trainer", gpu_count=8, finished_at=0),
    }
    assert check(known, 100, [1, 0], 4) is None
    known["trainer"]["finished_at"] = 120
    assert check(known, 100, [1, 1], 4) == "trainer"
    assert check(known, 100, [2, 2], 4) is None


def test_supplementary_pair_rejects_changed_work_or_hardware(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    check = runpy.run_path(str(BENCHMARK / "wave_report.py"))["matched_pair"]
    sources = {"router.py": "policy-hash", "replay.py": "harness-hash"}
    frozen = dict(source_sha256=sources)
    raw = dict(
        pool="same-pool",
        workload_sha256="same-workload",
        trace_sha256="same-trace",
        config=dict(source_sha256=sources, work_rates={"decode": 1000}),
        baseline=[dict(replica=str(i)) for i in range(8)],
        final=[dict(replica=str(i)) for i in range(8)],
    )
    check(raw, deepcopy(raw), frozen)
    changed = deepcopy(raw)
    changed["final"][0]["replica"] = "replacement"
    with pytest.raises(ValueError, match="Physical replicas changed"):
        check(raw, changed, frozen)
    changed = deepcopy(raw)
    changed["config"]["work_rates"]["decode"] = 2000
    with pytest.raises(ValueError, match="Unmatched work_rates"):
        check(raw, changed, frozen)
    changed = deepcopy(raw)
    changed["trace_sha256"] = "different-trace"
    with pytest.raises(ValueError, match="Unmatched trace_sha256"):
        check(raw, changed, frozen)


def test_resource_audit_deduplicates_and_does_not_count_queued_time(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    summarize = runpy.run_path(str(BENCHMARK / "resource_report.py"))["summarize"]
    task = dict(
        task_id="trainer", gpu_count=4, gpu_type="H100", started_at=100, finished_at=0
    )
    first, last = tmp_path / "first.json", tmp_path / "last.json"
    first.write_text(json.dumps([task | {"started_at": 0}]))
    last.write_text(
        json.dumps(
            [
                task | {"finished_at": 3700},
                task | {"task_id": "queued", "started_at": 0},
            ]
        )
    )
    result = summarize([first, last], 4000)
    assert result["gpu_hours"] == 4
    assert result["peak_gpus"] == 4
    assert result["observed_active_gpus"] == 0
    assert result["tasks_without_recorded_start"] == ["queued"]


def test_preparation_retry_drains_replica_and_remains_bounded(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    scope = runpy.run_path(str(BENCHMARK / "cache_replay.py"))
    monkeypatch.setattr(scope["asyncio"], "sleep", AsyncMock())
    setup = scope["setup_generate"]
    node = SimpleNamespace(upstream="upstream", replica_id="replica")
    request = httpx.Request("POST", "https://test/generate")
    states = [
        httpx.Response(
            200,
            request=request,
            json=dict(replica="replica", active=1, work=dict(request_ids=["old"])),
        ),
        httpx.Response(
            200,
            request=request,
            json=dict(replica="replica", active=0, work=dict(request_ids=[])),
        ),
    ]
    client = SimpleNamespace(
        post=AsyncMock(
            side_effect=[
                httpx.Response(502, request=request),
                httpx.Response(200, request=request),
            ]
        ),
        get=AsyncMock(side_effect=states),
    )
    retries = []
    asyncio.run(setup(client, "https://test", node, {}, retries))
    assert client.get.await_count == client.post.await_count == 2
    assert [r["status"] for r in retries] == [502]
    client.post = AsyncMock(return_value=httpx.Response(502, request=request))
    client.get = AsyncMock(return_value=states[-1])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(setup(client, "https://test", node, {}, []))
    assert client.post.await_count == 3
