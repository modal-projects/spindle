"""CPU regressions for the GPU gate; these tests never provision Modal apps."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from modal_proto import api_pb2

from scripts.gpu_ci import cleanup
from scripts.gpu_ci import run as runner
from scripts.gpu_ci.cleanup import owned_apps, should_stop
from scripts.gpu_ci.diagnostics import analyze
from scripts.gpu_ci.run import prepare
from scripts.gpu_ci.validate import (
    validate_client,
    validate_topology,
    report,
    compare_performance,
)
from scripts.rl_example import answer, datum
from spindle.deployment_cli import compile_configs

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = json.loads((ROOT / "tests/gpu/dapo_expected.json").read_text())


def fixture(model="adapter-a"):
    cfg = json.loads((ROOT / "scripts/rl_configs/dapo.json").read_text())
    cfg["updates"] = 3
    rows = [
        dict(event="client_created", model_id=model),
        dict(event="ready", time=1),
        dict(event="training_started", time=2),
    ]
    for u in range(4):
        rows.append(
            dict(
                event="publish",
                update=u,
                path=f"tinker://{model}:train:0/sampler_weights/{u}",
                seconds=1,
            )
        )
    for u in range(1, 4):
        rows += [
            dict(
                event="rollout_done",
                update=u,
                groups_sampled=8,
                groups_filtered=0,
                output_lengths=[2] * 64,
                output_tokens=128,
                used_output_tokens=128,
                reward=0.5,
                mixed_groups=8,
                truncated=0,
                seconds=1,
                time=u + 3,
            ),
            dict(event="train_start", update=u, policy_lag=0),
            dict(
                event="train_done",
                update=u,
                metrics={"loss:mean": 0.1},
                optimizer_metrics={"update_successful:mean": 1, "grad_norm:mean": 0.5},
                seconds=1,
            ),
            dict(event="update_done", update=u, seconds=2),
        ]
    rows += [
        dict(event="policy_probe", max_logprob_change=0.01),
        dict(event="completed", updates=3),
    ]
    manifest = dict(
        config=cfg,
        config_sha256="config",
        prompts_sha256="prompts",
        dataset_sha256="dataset",
        stop_token_ids=[1],
        packages={"tinker": "0.24.1"},
    )
    return rows, manifest, cfg


def test_valid_client():
    rows, m, cfg = fixture()
    assert validate_client(rows, m, cfg, EXPECTED) == "adapter-a"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("nan", "Nonfinite"),
        ("failed_optimizer", "Optimizer"),
        ("lag", "Policy lag"),
        ("duplicate_update", "Missing/duplicate"),
        ("wrong_adapter", "another adapter"),
        ("constant_rewards", "inconclusive"),
        ("no_gradient", "gradient"),
        ("unchanged", "did not change"),
        ("token_count", "token mismatch"),
        ("missing_publication", "publication"),
    ],
)
def test_regressions_fail(mutation, error):
    rows, m, cfg = fixture()
    train = next(r for r in rows if r["event"] == "train_done")
    if mutation == "nan":
        train["metrics"]["loss:mean"] = float("nan")
    elif mutation == "failed_optimizer":
        train["optimizer_metrics"]["update_successful:mean"] = 0
    elif mutation == "lag":
        next(r for r in rows if r["event"] == "train_start")["policy_lag"] = 2
    elif mutation == "duplicate_update":
        rows.append(next(r for r in rows if r["event"] == "update_done"))
    elif mutation == "wrong_adapter":
        next(r for r in rows if r["event"] == "publish")["path"] = (
            "tinker://other/sampler_weights/0"
        )
    elif mutation == "constant_rewards":
        for r in rows:
            if r["event"] == "rollout_done":
                r["mixed_groups"] = 0
    elif mutation == "no_gradient":
        for r in rows:
            if r["event"] == "train_done":
                r["optimizer_metrics"]["grad_norm:mean"] = 0
    elif mutation == "unchanged":
        next(r for r in rows if r["event"] == "policy_probe")["max_logprob_change"] = 0
    elif mutation == "token_count":
        next(r for r in rows if r["event"] == "rollout_done")["used_output_tokens"] = (
            127
        )
    elif mutation == "missing_publication":
        rows.remove(next(r for r in rows if r["event"] == "publish"))
    with pytest.raises(ValueError, match=error):
        validate_client(rows, m, cfg, EXPECTED)


def topology():
    return dict(
        placements={
            f"adapter-{i}": dict(engine_instance_id="one", engine_boot_id="boot")
            for i in range(8)
        },
        tasks=[dict(role="trainer", gpu_count=8, finished_at=0, gpu_type="H200")]
        + [
            dict(role="lora", gpu_count=2, finished_at=0, gpu_type="H200")
            for _ in range(8)
        ],
    )


def test_topology_rejects_multiple_trainers_or_missing_replicas():
    snap = topology()
    validate_topology(snap, list(snap["placements"]), EXPECTED)
    bad = copy.deepcopy(snap)
    bad["placements"]["adapter-0"]["engine_instance_id"] = "two"
    with pytest.raises(ValueError, match="one trainer"):
        validate_topology(bad, list(snap["placements"]), EXPECTED)
    bad = copy.deepcopy(snap)
    bad["tasks"].pop()
    with pytest.raises(ValueError, match="inference allocation"):
        validate_topology(bad, list(snap["placements"]), EXPECTED)


def test_cleanup_only_owns_exact_ci_names_and_expiry():
    run = "spindle-ci-dapo-1790970000-0123456789ab"
    for app in owned_apps(run):
        assert not should_stop(app, 1790969999)
        assert should_stop(app, 1790970000)
        assert should_stop(app, 0, run)
    for app in [
        "spindle",
        "spindle-trainer-production",
        run + "-extra",
        "spindle-gpu-ci-janitor",
        "spindle-ci-dapo-1790970000-wrong",
    ]:
        assert not should_stop(app, 9999999999)
        assert not should_stop(app, 0, run)
    with pytest.raises(ValueError):
        owned_apps("spindle")


def test_prepare_keeps_workload_and_resolves_recipe(tmp_path):
    for mode in ["correctness", "performance"]:
        output = tmp_path / mode
        prepare(output, mode, "ci-test", record_reference=(mode == "performance"))
        plan = json.loads((output / "plan.json").read_text())
        spec = importlib.util.spec_from_file_location(
            "ci_recipe", output / "deployment.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        recipe = mod.config
        compiled = compile_configs([str(output / "deployment.py")])
        assert compiled[0].name == plan["name"]
        assert recipe.name == plan["name"]
        assert recipe.platform["frontend"] == plan["name"]
        assert recipe.platform["modal"]["environment"] == "ci-test"
        assert recipe.trainer_gpus_per_node == 8
        assert recipe.inference_gpus_per_node * recipe.inference_max_replicas == 16
        assert recipe.miles_cfg["max_lora_slots"] == 8
        cfg = json.loads((output / "config.json").read_text())
        original = json.loads((ROOT / "scripts/rl_configs/dapo.json").read_text())
        assert {k: v for k, v in cfg.items() if k != "updates"} == {
            k: v for k, v in original.items() if k != "updates"
        }
        assert cfg["updates"] == (3 if mode == "correctness" else 30)


def test_report_enforces_barrier_and_unique_models(tmp_path):
    plan = dict(mode="correctness", commit="test", dataset_sha256="dataset")
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "expected.json").write_text(json.dumps(EXPECTED))
    snap = topology()
    (tmp_path / "topology.json").write_text(json.dumps(snap))
    for i in range(8):
        rows, m, cfg = fixture(f"adapter-{i}")
        p = tmp_path / f"client-{i:02d}"
        p.mkdir()
        (p / "manifest.json").write_text(json.dumps(m))
        (p / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    summary = report(tmp_path)
    assert summary["clients"] == 8
    assert summary["aggregate_rollout_tps"] == 1024
    p = tmp_path / "client-07/events.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    next(r for r in rows if r["event"] == "ready")["time"] = 10
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="barrier"):
        report(tmp_path)


def test_grader_and_training_payload():
    assert answer(r"answer: \boxed{4}") == 4
    assert answer("nonsense") is None
    d = datum([1, 2], [3, 4], [-0.2, -0.3], 1.0, 2)
    assert d.model_input.to_ints() == [1, 2, 3]
    with pytest.raises(ValueError, match="alignment|behavior"):
        datum([1], [2, 3], [-0.1], 1.0, 1)


def test_cleanup_attempts_remaining_apps_after_one_stop_fails(monkeypatch):
    run = "spindle-ci-dapo-1790970000-0123456789ab"
    apps = [
        SimpleNamespace(
            app_id=str(i), description=name, state=api_pb2.APP_STATE_DEPLOYED
        )
        for i, name in enumerate(owned_apps(run))
    ]
    apps.append(
        SimpleNamespace(
            app_id="production", description="spindle", state=api_pb2.APP_STATE_DEPLOYED
        )
    )
    stopped = []

    class Stub:
        async def AppList(self, request):
            assert request.environment_name == "ci-test"
            return SimpleNamespace(apps=apps)

        async def AppStop(self, request):
            stopped.append(request.app_id)
            if request.app_id == "0":
                raise RuntimeError("transient stop failure")

    async def client():
        return SimpleNamespace(stub=Stub())

    monkeypatch.setattr(cleanup._Client, "from_env", client)
    with pytest.raises(RuntimeError, match="transient"):
        asyncio.run(cleanup.stop_apps("ci-test", run))
    assert stopped == ["0", "1", "2", "3"]


def test_diagnostics_does_not_invent_missing_queue_time(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"name": "example"}))
    p = tmp_path / "client-00"
    p.mkdir()
    rows = [
        dict(event="client_created", model_id="a"),
        dict(event="training_started", time=1),
        dict(event="training_finished", time=100),
    ]
    (p / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = analyze(tmp_path)
    assert result["status"] == "missing_backend_logs"
    assert result["backend_call_occupancy"] is None
    assert result["training_queue_mean_s"] is None


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_deployment_failure_still_cleans_up_and_records_failure(
    tmp_path, monkeypatch, cleanup_fails
):
    output = tmp_path / "run"
    prepare(output, "correctness", "ci-test")
    monkeypatch.setenv("TINKER_API_KEY", "tml-test-key")
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    cleaned = []

    def fail_deployment(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "modal deploy")

    def stop(path):
        cleaned.append(path)
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    async def no_resources(path):
        return None

    monkeypatch.setattr(runner.subprocess, "run", fail_deployment)
    monkeypatch.setattr(runner, "cleanup", stop)
    monkeypatch.setattr(runner, "collect_logs", lambda *args: None)
    monkeypatch.setattr(runner, "finalized_resources", no_resources)
    monkeypatch.setattr(runner, "analyze", lambda *args: None)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        runner.run(output)
    assert cleaned == [output]
    assert json.loads((output / "status.json").read_text())["state"] == "failed"
    assert "CalledProcessError" in (output / "failure.json").read_text()
    assert not (output / "reference.json").exists()


def reference_fixture(value=100):
    return dict(
        schema=1,
        metric="end_to_end_output_tps",
        value=value,
        commit="known-good",
        run_name="reference-run",
        identity=dict(run={"config": "fixed"}, client={"prompts": "fixed"}),
    )


@pytest.mark.parametrize(
    "value,status", [(100, "passed"), (85, "passed"), (84.99, "failed")]
)
def test_throughput_regression_boundary(value, status):
    reference = reference_fixture()
    result = compare_performance(value, reference["identity"], reference)
    assert result["status"] == status
    assert result["minimum_value"] == 85


def test_reference_rejects_incomparable_runs():
    reference = reference_fixture()
    identity = copy.deepcopy(reference["identity"])
    identity["run"]["config"] = "different"
    with pytest.raises(ValueError, match="workload/topology"):
        compare_performance(100, identity, reference)
    identity = copy.deepcopy(reference["identity"])
    identity["client"]["prompts"] = "different"
    with pytest.raises(ValueError, match="prompts/packages"):
        compare_performance(100, identity, reference)
    reference["value"] = float("nan")
    with pytest.raises(ValueError, match="Invalid reference"):
        compare_performance(100, identity, reference)


def test_performance_requires_reference_before_deploy(tmp_path):
    with pytest.raises(ValueError, match="Missing performance reference"):
        prepare(
            tmp_path / "run",
            "performance",
            "ci-test",
            reference=tmp_path / "missing.json",
        )
    assert not (tmp_path / "run/plan.json").exists()
    with pytest.raises(ValueError, match="full performance"):
        prepare(tmp_path / "short", "correctness", "ci-test", record_reference=True)


def test_performance_report_records_candidate_and_fails_regression(tmp_path):
    prepare(tmp_path / "run", "performance", "ci-test", record_reference=True)
    root = tmp_path / "run"
    cfg = json.loads((root / "config.json").read_text())
    (root / "topology.json").write_text(json.dumps(topology()))
    plan = json.loads((root / "plan.json").read_text())
    for i in range(8):
        rows, manifest, _ = fixture(f"adapter-{i}")
        # Expand the synthetic trace to all 30 updates.
        for u in range(4, 31):
            for event in (
                "rollout_done",
                "train_start",
                "train_done",
                "update_done",
                "publish",
            ):
                row = copy.deepcopy(
                    next(r for r in rows if r["event"] == event and r["update"] == 3)
                )
                row["update"] = u
                if event == "publish":
                    row["path"] = f"tinker://adapter-{i}:train:0/sampler_weights/{u}"
                if "time" in row:
                    row["time"] = u + 3
                rows.append(row)
        next(r for r in rows if r["event"] == "completed")["updates"] = 30
        rows.append(dict(event="training_finished", time=40))
        manifest["config"] = cfg
        manifest["dataset_sha256"] = plan["dataset_sha256"]
        p = root / f"client-{i:02d}"
        p.mkdir()
        (p / "manifest.json").write_text(json.dumps(manifest))
        (p / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = report(root)
    assert summary["performance"]["status"] == "reference_candidate"
    assert summary["end_to_end_output_tps"] == 8 * 30 * 128 / 38
    reference = reference_fixture(summary["end_to_end_output_tps"] * 2)
    reference["identity"] = summary["benchmark_identity"]
    (root / "baseline.json").write_text(json.dumps(reference))
    prepare(
        tmp_path / "comparison",
        "performance",
        "ci-test",
        reference=root / "baseline.json",
    )
    reference["identity"]["run"]["stagger_seconds"] = 0
    (root / "incompatible.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="workload/topology"):
        prepare(
            tmp_path / "incompatible",
            "performance",
            "ci-test",
            reference=root / "incompatible.json",
        )
    plan["record_reference"] = False
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Performance regression"):
        report(root)
    assert json.loads((root / "summary.json").read_text())["status"] == "failed"
    assert "**failed**" in (root / "summary.md").read_text()
