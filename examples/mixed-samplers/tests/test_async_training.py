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


def test_sampling_starts_after_the_first_ready_sampler():
    async def run():
        async def fast():
            return {"tokens_per_second": 5.0}

        async def slow():
            await asyncio.sleep(0.05)
            return {"tokens_per_second": 2.0}

        async def down():
            await asyncio.sleep(0.05)
            raise RuntimeError("no upstreams")

        pool = training_module.ReadySamplers()
        await pool.open({"fast": fast(), "slow": slow(), "down": down()})
        assert pool.groups == ["fast"]
        assert pool.weights == [5.0]
        assert set(pool.pending_names()) == {"slow", "down"}
        await pool.adopt(lambda: None)
        assert pool.groups == ["fast", "slow"]
        assert pool.weights == [5.0, 2.0]
        assert "down" in pool.failures
        assert pool.pending_names() == []

        empty = training_module.ReadySamplers()
        with pytest.raises(RuntimeError, match="no samplers became ready"):
            await empty.open({"down": down()})

    asyncio.run(run())


def test_cancelling_startup_stops_pending_measurements():
    async def run():
        tasks = []

        async def never():
            tasks.append(asyncio.current_task())
            await asyncio.Event().wait()

        pool = training_module.ReadySamplers()
        opening = asyncio.create_task(pool.open({"a": never(), "b": never()}))
        await asyncio.wait_for(_until(lambda: len(tasks) == 2), timeout=1)
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert pool.pending_names() == []
        assert tasks and all(task.cancelled() for task in tasks)

    asyncio.run(run())


def test_a_cancelled_measurement_does_not_stop_the_other_groups():
    async def run():
        async def ready():
            return {"tokens_per_second": 1.0}

        async def stay():
            await asyncio.Event().wait()

        pool = training_module.ReadySamplers()
        await pool.open({"fast": ready(), "later": stay()})
        next(iter(pool._pending)).cancel()
        await pool.adopt(lambda: None)
        assert pool.groups == ["fast"]
        assert "later" in pool.failures

    asyncio.run(run())


def test_training_finishes_while_a_sampler_measurement_is_pending():
    async def run():
        hung = []

        async def ready():
            return {"tokens_per_second": 1.0}

        async def stay():
            hung.append(asyncio.current_task())
            await asyncio.Event().wait()

        pool = training_module.ReadySamplers()
        await pool.open({"fast": ready(), "queued": stay()})

        async def training():
            await asyncio.sleep(0.01)

        await asyncio.wait_for(
            training_module.run_until_training_finishes(
                training(), pool.adopt(lambda: None)
            ),
            timeout=1,
        )
        assert hung and hung[0].cancelled()
        assert pool.pending_names() == []

    asyncio.run(run())


def test_train_assigns_a_sampler_that_joins_mid_run(tmp_path, monkeypatch):
    monkeypatch.setattr(training_module, "PROMPTS_PER_UPDATE", 1)
    monkeypatch.setattr(training_module, "TRAJECTORIES_PER_PROMPT", 1)
    seen = []
    pool = training_module.ReadySamplers()
    pool._admit("fast", {"tokens_per_second": 1.0}, 1.0)

    async def sample(task, gateway, group, model_id, version, seed):
        seen.append(group)
        if "slow" not in pool.groups:
            pool._admit("slow", {"tokens_per_second": 9.0}, 9.0)
            pool.weights[0] = 0.0
        row = {
            "instance_id": task["instance_id"],
            "tokens": [1, 2],
            "active": [0, 1],
            "logq": [0.0, 0.0],
            "offsets": [0, 0, 1],
            "support": [2],
            "support_logq": [0.0],
            "reward": 1.0,
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
            return self.result

    class Training:
        model_id = "test"

        def forward_backward(self, data, loss, config):
            return Future(SimpleNamespace(metrics={"entropy:mean": 0.0}))

        def optim_step(self, params):
            return Future(SimpleNamespace(metrics={"update_successful:mean": 1.0}))

        def save_weights_for_sampler(self, name):
            return Future(SimpleNamespace(path=name))

        def save_state(self, name):
            return Future(SimpleNamespace(path=name))

    asyncio.run(
        training_module.train(
            Training(),
            {"fast": "http://fast", "slow": "http://slow"},
            pool.weights,
            SimpleNamespace(
                output=str(tmp_path), inflight_prompts=1, start_step=0, steps=2
            ),
            [{"instance_id": "one"}],
            rollout_worker=SimpleNamespace(remote=SimpleNamespace(aio=sample)),
            sampler_pool=pool,
        )
    )
    assert seen[0] == "fast"
    assert "slow" in seen


def test_require_all_samplers_stops_for_a_failed_or_missing_group():
    pool = training_module.ReadySamplers()
    pool._admit("fast", {"tokens_per_second": 1.0}, 1.0)
    args = SimpleNamespace(
        require_all_samplers=True, sampler_join_steps=1, start_step=0
    )
    assert training_module.sampler_join_error(pool, args, step=1) is None
    pool.failures["down"] = "RuntimeError('no upstreams')"
    assert "failed" in training_module.sampler_join_error(pool, args, step=1)
    pool.failures.clear()
    pool._pending[object()] = "queued"
    assert training_module.sampler_join_error(pool, args, step=1) is None
    assert "pending" in training_module.sampler_join_error(pool, args, step=2)
    pool._pending.clear()
    pool._pending[object()] = "queued"
    assert "never joined" in training_module.sampler_join_error(
        pool, args, finished=True
    )


async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0.01)


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


@pytest.mark.parametrize("rollout_concurrency", [None, 2])
@pytest.mark.parametrize("published_versions", [None, (7, 19)])
@pytest.mark.parametrize(
    ("cached_version", "cached_event", "start_step"),
    [
        (None, None, 0),
        (1, "trajectory_completed", 0),
        (1, "trajectory_restored", 0),
        (2, "trajectory_completed", 0),
        (2, "trajectory_restored", 0),
        (1, "trajectory_completed", 1),
        (1, "trajectory_restored", 1),
        (2, "trajectory_completed", 1),
    ],
)
def test_producers_continue_during_native_training_and_use_old_rollouts(
    tmp_path,
    monkeypatch,
    cached_version,
    cached_event,
    start_step,
    published_versions,
    rollout_concurrency,
):
    monkeypatch.setattr(training_module, "PROMPTS_PER_UPDATE", 2)
    monkeypatch.setattr(training_module, "TRAJECTORIES_PER_PROMPT", 2)

    sampled_seeds = []
    requested_versions = []
    active_calls = peak_calls = 0

    async def sample(task, gateway, group, model_id, version, seed):
        nonlocal active_calls, peak_calls
        sampled_seeds.append(seed)
        requested_versions.append(version)
        active_calls += 1
        peak_calls = max(peak_calls, active_calls)
        try:
            await asyncio.sleep(0.01)
        finally:
            active_calls -= 1
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
        output=str(tmp_path), inflight_prompts=3, start_step=start_step, steps=2
    )
    checkpoint = "tinker://prior/weights/step-00001" if start_step else None
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
                    "arguments": {"checkpoint": checkpoint, "start_step": start_step},
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
            previous,
            [{"instance_id": "one"}],
            checkpoint=checkpoint,
            start_step=start_step,
        )
        if checkpoint:
            for target, step in (
                (None, 0),
                ("tinker://other/weights/step-00001", 1),
                (checkpoint, 2),
            ):
                with pytest.raises(
                    ValueError, match="different initial training state"
                ):
                    training_module.initial_rollout_cache(
                        previous,
                        [{"instance_id": "one"}],
                        checkpoint=target,
                        start_step=step,
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
    publications = iter(published_versions or ())

    async def version_resolver():
        return next(publications)

    asyncio.run(
        training_module.train(
            Training(),
            {"b200-nvfp4": "unused"},
            [1.0],
            args,
            [{"instance_id": "one"}],
            cache=cache,
            version_resolver=version_resolver if published_versions else None,
            rollout_concurrency=rollout_concurrency,
        )
    )
    if rollout_concurrency is not None:
        assert peak_calls == rollout_concurrency
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
    assert finishes[1]["version"] == (
        published_versions[-1] if published_versions else 3
    )
    assert finishes[1]["groups"]["b200-nvfp4"]["policy_age_mean"] == 1
    if published_versions:
        assert published_versions[0] in requested_versions
    # The second update consumed the still-valid version-1 samples.
    assert all(
        e["versions"][0]["start"] == 1
        for e in events
        if e["event"] == "trajectory_completed"
    )
