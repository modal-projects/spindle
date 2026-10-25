import asyncio
import gzip
import importlib
import json
import pickle
import time
from types import SimpleNamespace

import pytest

training_module = importlib.import_module("mixed_samplers.train")


def test_token_diagnostics_reconstruct_means_after_sdk_chunk_reduction():
    from tinker import ForwardBackwardOutput
    from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

    chunks = [
        ForwardBackwardOutput(
            loss_fn_output_type="MegatronScoreCenteredMISLoss",
            loss_fn_outputs=[{}],
            metrics={
                "tokens:sum": count,
                **{
                    f"{name}:sum": count * value for name in training_module.DIAGNOSTICS
                },
                **{f"{name}:mean": value for name in training_module.DIAGNOSTICS},
            },
        )
        for count, value in [(1, 0.1), (9, 0.9)]
    ]
    combined = combine_fwd_bwd_output_results(chunks)
    metrics, aggregation = training_module.diagnostic_means(combined.metrics)
    assert aggregation == "token_mean"
    for name in training_module.DIAGNOSTICS:
        assert metrics[f"{name}:mean"] == pytest.approx(0.82)
    legacy = {k: v for k, v in combined.metrics.items() if not k.endswith(":sum")}
    assert training_module.diagnostic_means(legacy) == (legacy, "sdk_chunk_means")


@pytest.mark.parametrize("validation_fails", [False, True])
def test_recovery_trains_while_validating_and_cancels_on_failure(validation_fails):
    async def run():
        started, validated, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def train():
            started.set()
            try:
                await validated.wait()
            finally:
                stopped.set()

        async def validate():
            await started.wait()
            if validation_fails:
                raise RuntimeError("sampler failed validation")
            validated.set()

        operation = training_module.run_with_sampler_validation(train(), validate())
        if validation_fails:
            with pytest.raises(ExceptionGroup) as exc:
                await asyncio.wait_for(operation, timeout=1)
            assert str(exc.value.exceptions[0]) == "sampler failed validation"
        else:
            await asyncio.wait_for(operation, timeout=1)
        assert started.is_set() and stopped.is_set()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("cached_version", "cached_event"),
    [
        (None, None),
        (1, "trajectory_completed"),
        (1, "trajectory_restored"),
        (2, "trajectory_completed"),
        (2, "trajectory_restored"),
    ],
)
def test_producers_continue_during_native_training_and_use_old_rollouts(
    tmp_path, monkeypatch, cached_version, cached_event
):
    monkeypatch.setattr(training_module, "PROMPTS_PER_UPDATE", 2)
    monkeypatch.setattr(training_module, "TRAJECTORIES_PER_PROMPT", 2)

    sampled_seeds = []

    async def sample(task, gateway, group, model_id, version, seed):
        sampled_seeds.append(seed)
        await asyncio.sleep(0.01)
        row = {
            "instance_id": task["instance_id"],
            "tokens": [1, 2],
            "active": [0, 1],
            "logq": [0.0, 0.0],
            "offsets": [0, 0, 1],
            "support": [2],
            "support_logq": [0.0],
            "reward": float(seed % 3 == 0),
            "format_error": False,
            "generated_tokens": 1,
            "versions": [{"start": 1, "end": 1}],
            "group": group,
            "started_at": time.time(),
            "finished_at": time.time(),
            "stop_reason": "submit",
            "turns": 1,
        }
        return gzip.compress(pickle.dumps(row))

    monkeypatch.setattr(
        training_module,
        "sample_trajectory",
        SimpleNamespace(remote=SimpleNamespace(aio=sample)),
    )

    class Future:
        def __init__(self, result):
            self.result = result

        async def result_async(self):
            await asyncio.sleep(0.03)
            return self.result

    class Training:
        model_id = "test"

        def forward_backward(self, data, loss, config):
            assert loss == "score_centered_mis"
            assert len(data) == 2
            return Future(SimpleNamespace(metrics={"entropy:mean": 0.0}))

        def optim_step(self, params):
            return Future(SimpleNamespace(metrics={"update_successful:mean": 1.0}))

        def save_weights_for_sampler(self, name):
            return Future(SimpleNamespace(path=name))

        def save_state(self, name):
            return Future(SimpleNamespace(path=name))

    args = SimpleNamespace(
        output=str(tmp_path), inflight_prompts=3, start_step=0, steps=2
    )
    cache = None
    if cached_version is not None:
        previous = tmp_path / "previous"
        (previous / "trajectories").mkdir(parents=True)
        row = pickle.loads(
            gzip.decompress(
                asyncio.run(
                    sample(
                        {"instance_id": "one"},
                        "unused",
                        "b200-nvfp4",
                        "old-model",
                        1,
                        42,
                    )
                )
            )
        )
        row["versions"] = [{"start": cached_version, "end": cached_version}]
        (previous / "trajectories" / "00000000-0.pkl.gz").write_bytes(
            gzip.compress(pickle.dumps(row))
        )
        (previous / "manifest.json").write_text(
            json.dumps(
                {
                    "model": training_module.config.model,
                    "model_revision": training_module.config.model_revision,
                    "dataset": training_module.DATASET,
                    "dataset_revision": training_module.DATASET_REVISION,
                    "groups": training_module.GROUPS,
                    "capacity": {
                        g["name"]: {"tokens_per_second": 1.0}
                        for g in training_module.GROUPS
                    },
                    "top_p": training_module.TOP_P,
                    "temperature": training_module.TEMPERATURE,
                    "batch": [2, 2],
                    "arguments": {"checkpoint": None, "start_step": 0},
                }
            )
        )
        (previous / "events.jsonl").write_text(
            json.dumps(
                {
                    "event": "prompt_started",
                    "prompt": 0,
                    "instance_id": "one",
                    "groups": ["b200-nvfp4"] * 2,
                }
            )
            + "\n"
            + json.dumps({"event": cached_event, "prompt": 0, "index": 0})
            + "\n"
        )
        cache = training_module.initial_rollout_cache(
            previous, [{"instance_id": "one"}]
        )
        sampled_seeds.clear()
    if cached_version == 2:
        with pytest.raises(ValueError, match="not from the initial policy"):
            asyncio.run(
                training_module.train(
                    Training(),
                    {"b200-nvfp4": "unused"},
                    [1.0],
                    args,
                    [{"instance_id": "one"}],
                    cache=cache,
                )
            )
        return
    asyncio.run(
        training_module.train(
            Training(),
            {"b200-nvfp4": "unused"},
            [1.0],
            args,
            [{"instance_id": "one"}],
            cache=cache,
        )
    )
    if cache is not None:
        assert 42 not in sampled_seeds
        assert 1042 in sampled_seeds
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    starts = [e for e in events if e["event"] == "training_started"]
    finishes = [e for e in events if e["event"] == "training_completed"]
    assert len(finishes) == 2
    assert any(
        starts[0]["at"] < e["at"] < finishes[0]["at"]
        for e in events
        if e["event"] == "prompt_started"
    )
    assert finishes[1]["version"] == 3
    # The second update consumed the still-valid version-1 samples.
    assert all(
        e["versions"][0]["start"] == 1
        for e in events
        if e["event"] == "trajectory_completed"
    )
