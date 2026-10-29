import asyncio
import json
from types import SimpleNamespace

import pytest
from mixed_samplers import train_miles
from test_miles_cache import make_cache


def saved_run(path, steps=(3, 4)):
    manifest = make_cache(path, [{"instance_id": "task"}])
    manifest.update(
        lr=train_miles.LEARNING_RATE,
        context_length=train_miles.config.max_context_length,
    )
    (path / "manifest.json").write_text(json.dumps(manifest))
    for index, step in enumerate(steps):
        (path / f"client-{index}" / "checkpoint.json").write_text(
            json.dumps({"step": step, "path": f"tinker://model-{index}/weights/saved"})
        )
    return manifest


@pytest.mark.parametrize("steps", [(3, 4), (5, 3)])
def test_resume_keeps_independent_steps_and_original_capacity(tmp_path, steps):
    manifest = saved_run(tmp_path, steps)
    resume = train_miles.checkpoint_resume(tmp_path, 2, 5)
    assert [c["step"] for c in resume["checkpoints"]] == list(steps)
    assert resume["capacity"] == manifest["capacity"]


@pytest.mark.parametrize("steps", [(0, 1), (6, 3), (True, 1), (5, 5)])
def test_resume_rejects_invalid_or_completed_targets(tmp_path, steps):
    saved_run(tmp_path, steps)
    with pytest.raises(ValueError):
        train_miles.checkpoint_resume(tmp_path, 2, 5)


def test_resume_rejects_changed_training_settings_and_missing_checkpoint(tmp_path):
    manifest = saved_run(tmp_path)
    manifest["lr"] *= 10
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training settings"):
        train_miles.checkpoint_resume(tmp_path, 2, 5)
    manifest["lr"] = train_miles.LEARNING_RATE
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "client-1/checkpoint.json").unlink()
    with pytest.raises(FileNotFoundError):
        train_miles.checkpoint_resume(tmp_path, 2, 5)


@pytest.mark.parametrize("steps", [(3, 4), (5, 3)])
def test_resume_loads_optimizer_before_publication_and_runs_clients_independently(
    tmp_path, monkeypatch, steps
):
    old = tmp_path / "previous"
    old.mkdir()
    manifest = saved_run(old, steps)
    resume = train_miles.checkpoint_resume(old, 2, 5)
    calls, clients, trained = [], [], []

    def future(call, value=None):
        async def result():
            calls.append(call)
            await asyncio.sleep(0)
            return value

        return SimpleNamespace(result_async=result)

    class Training:
        def __init__(self, index):
            self.index, self.model_id = index, f"new-{index}"

        def load_state_with_optimizer(self, path):
            return future(("load_optimizer", self.index, path))

        def save_weights_for_sampler(self, name):
            return future(("publish", self.index, name), SimpleNamespace(path=name))

    async def create(*args, **kwargs):
        client = Training(len(clients))
        clients.append(client)
        return client

    async def latest(model_id, bulletin):
        return SimpleNamespace(version=11 + int(model_id[-1]))

    async def measure(*args):
        return {"tokens_per_second": 999.0}

    class Http:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, headers, json):
            calls.append(("unload", json["model_id"]))
            return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(train_miles.httpx, "AsyncClient", Http)
    monkeypatch.setattr(train_miles, "latest_ref", latest)
    monkeypatch.setattr(
        train_miles,
        "measure_capacity",
        SimpleNamespace(remote=SimpleNamespace(aio=measure)),
    )
    args = SimpleNamespace(
        output=tmp_path / "resumed",
        clients=2,
        steps=5,
        stagger_seconds=0.01,
        trainer_validation=None,
        validation_adapters=None,
        initial_rollout_cache=None,
        resume_from=old,
        rollout_concurrency=512,
        inflight_prompts=192,
    )
    gateways = {g["name"]: g["name"] for g in train_miles.GROUPS}

    async def run():
        second_started = asyncio.Event()

        async def train(client, gateways, weights, client_args, tasks, **kwargs):
            index = client.index
            assert client_args.start_step == steps[index]
            assert client_args.steps == 5 - steps[index]
            assert kwargs["initial_version"] == 11 + index
            assert kwargs["cache"] is None
            assert weights == [
                r["tokens_per_second"] for r in manifest["capacity"].values()
            ]
            assert await kwargs["version_resolver"]() == 11 + index
            trained.append(index)
            if index == 0:
                await asyncio.wait_for(second_started.wait(), 1)
            else:
                second_started.set()

        monkeypatch.setattr(train_miles, "train", train)
        await train_miles.run_clients(
            args,
            SimpleNamespace(create_lora_training_client_async=create),
            gateways,
            [],
            "unused",
            "test-key",
            resume=resume,
        )

    asyncio.run(run())
    assert trained == [0, 1]
    for index in range(2):
        checkpoint = resume["checkpoints"][index]
        assert calls.index(("load_optimizer", index, checkpoint["path"])) < calls.index(
            ("publish", index, "initial")
        )
        assert ("unload", f"new-{index}") in calls
        assert (
            json.loads(
                (args.output / f"client-{index}" / "checkpoint.json").read_text()
            )
            == checkpoint
        )
    resumed = json.loads((args.output / "manifest.json").read_text())
    assert resumed["initial_steps"] == list(steps)
    assert resumed["initial_versions"] == [11, 12]
    assert resumed["capacity"] == manifest["capacity"]
    with pytest.raises(ValueError, match="untrained adapters"):
        train_miles.initial_client_rollouts(args.output, [], 2)
