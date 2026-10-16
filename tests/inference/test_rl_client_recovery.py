"""Fault injection against the real client loop without GPU allocation."""

import json
import gzip
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.benchmarks.lora_routing.recovery import (
    ExistingProcess,
    atomic_json,
    latest_commit,
    classify_failure,
)
from scripts.benchmarks.lora_routing.client import Client
from scripts.benchmarks.lora_routing import supervise


def future(value):
    return SimpleNamespace(result=lambda **kw: value)


def client(tmp_path, commit=None, fail_save=False):
    instance = Client.__new__(Client)
    instance.root = tmp_path
    instance.cfg = dict(
        updates=3,
        warmup_samples=0,
        clip_low=0.8,
        clip_high=1.28,
        learning_rate=1e-5,
        beta1=0.9,
        beta2=0.95,
        eps=1e-12,
        grad_clip_norm=1,
    )
    instance.commit = commit
    instance.completed_update = commit["update"] if commit else 0
    instance.recover = True
    instance.sampling_paths = {}
    instance.service = SimpleNamespace(
        create_sampling_client=lambda **kw: kw["model_path"]
    )
    instance.trainer = SimpleNamespace(
        forward_backward=lambda *a, **kw: future(SimpleNamespace(metrics={})),
        optim_step=lambda *a, **kw: future(SimpleNamespace(metrics={})),
        save_weights_for_sampler=lambda name: future(SimpleNamespace(path=name)),
        save_state=Mock(
            side_effect=RuntimeError("connection lost after optimizer")
            if fail_save
            else lambda name: future(SimpleNamespace(path=name))
        ),
    )
    instance.event = Mock()
    instance.unload = Mock()

    def publish(update, publication=None):
        instance.sampling_paths[update] = f"policy-{update}"
        return f"policy-{update}", update

    instance.publish = publish
    instance.rollout = Mock(
        side_effect=lambda snapshot, update: (
            [],
            dict(update=update, policy_version=snapshot[1], started_at=1),
        )
    )
    return instance


def test_commits_preserve_prefetch_policy_and_restart_step(tmp_path):
    first = client(tmp_path)
    first.run()
    commits = [
        json.loads(p.read_text()) for p in sorted(tmp_path.glob("commit-*.json"))
    ]
    assert [c["rollout"]["policy_version"] for c in commits] == [0, 0, 1]
    assert commits[1]["next_policy_version"] == 1
    (tmp_path / "commit-003.json").unlink()
    resumed = client(tmp_path, latest_commit(tmp_path))
    resumed.run()
    assert resumed.rollout.call_count == 1
    assert resumed.rollout.call_args.args == (("policy-1", 1), 3)
    assert latest_commit(tmp_path)["update"] == 3


def test_optimizer_success_checkpoint_failure_does_not_commit(tmp_path):
    failed = client(tmp_path, fail_save=True)
    with pytest.raises(RuntimeError, match="connection lost"):
        failed.run()
    assert latest_commit(tmp_path) is None
    assert not any(
        c.args and c.args[0] == "update_done" for c in failed.event.call_args_list
    )


def test_partial_commit_ignored_and_holes_rejected(tmp_path):
    atomic_json(tmp_path / "commit-001.json", {"update": 1})
    (tmp_path / "commit-002.json.tmp").write_text("incomplete")
    assert latest_commit(tmp_path) == {"update": 1}
    atomic_json(tmp_path / "commit-003.json", {"update": 3})
    with pytest.raises(ValueError, match="Noncontiguous"):
        latest_commit(tmp_path)


@pytest.mark.parametrize(
    "error,kind",
    [
        ("BackendFailed: trainer unavailable", "retry_from_checkpoint"),
        ("Trainer exited: serving", "retry_from_checkpoint"),
        ("ReadTimeout", "retry_from_checkpoint"),
        ("CUDA out of memory", "requires_fix"),
        ("Nonfinite training metrics", "requires_fix"),
        ("ValueError: invalid shape", "requires_fix"),
        ("unexpected unidentified failure", "requires_investigation"),
    ],
)
def test_retry_classification(error, kind):
    assert classify_failure(error) == kind


@pytest.mark.parametrize("code", [0, 7])
def test_adopted_child_exit_status(code):

    child = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
    observed = ExistingProcess(child.pid)
    try:
        deadline = time.monotonic() + 5
        while observed.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert observed.poll() == code
    finally:
        child.wait(timeout=5)


def test_committed_update_survives_missing_completion_event(tmp_path):
    first = client(tmp_path)

    def fail_after_commit(event, **fields):
        if event == "update_done":
            raise RuntimeError("process interrupted after durable commit")

    first.event = fail_after_commit
    with pytest.raises(RuntimeError, match="durable commit"):
        first.run()
    assert latest_commit(tmp_path)["update"] == 1
    resumed = client(tmp_path, latest_commit(tmp_path))
    resumed.run()
    assert [c.args[1] for c in resumed.rollout.call_args_list] == [2, 3]
    assert [c.args[0][1] for c in resumed.rollout.call_args_list] == [0, 1]


def test_saved_group_reused_without_resampling(tmp_path):

    instance = Client.__new__(Client)
    instance.root, instance.recover, instance.event = tmp_path, True, Mock()
    saved = dict(group=32, policy_version=0, sequences=["recorded"])
    with gzip.open(tmp_path / "group-000032.json.gz", "wt") as stream:
        json.dump(saved, stream)
    sampler = Mock()
    assert instance.group(32, (sampler, 0)) == saved
    sampler.sample.assert_not_called()
    with pytest.raises(ValueError, match="policy/group mismatch"):
        instance.group(32, (sampler, 1))


def test_supervisor_restarts_failed_client_and_preserves_peer(tmp_path, monkeypatch):

    config = tmp_path / "config.json"
    config.write_text('{"updates":1}')
    root = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervise",
            "--root",
            str(root),
            "--base-url",
            "http://unused",
            "--config",
            str(config),
            "--clients",
            "2",
            "--stagger-seconds",
            "0",
            "--recover-clients",
            "--keep-apps",
            "--owned-apps",
            "owned-test",
        ],
    )
    clock = [1000.0]
    monkeypatch.setattr(supervise.time, "time", lambda: clock[0])
    monkeypatch.setattr(supervise.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        supervise.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    calls = []

    class Process:
        def __init__(self, command, **kw):
            index = int(command[command.index("--client-index") + 1])
            calls.append(index)
            output = root / f"client-{index:02d}"
            output.mkdir(exist_ok=True)
            failed = calls.count(index) == 1 and index == 0
            self.pid, self.returncode = 100 + len(calls), int(failed)
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="ready")) + "\n")
                if not failed:
                    stream.write(json.dumps(dict(event="update_done", update=1)) + "\n")
            if failed:
                atomic_json(output / "failure.json", dict(error="ReadTimeout"))

        def poll(self):
            return self.returncode

        def wait(self, **kw):
            return self.returncode

    async def resources(*a, **kw):
        return {}

    monkeypatch.setattr(supervise.subprocess, "Popen", Process)
    monkeypatch.setattr(supervise, "resources", SimpleNamespace(aio=resources))
    supervise.main()
    assert calls == [0, 1, 0]
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["updates"] == [1, 1]
    assert len(status["recoveries"]) == 1
