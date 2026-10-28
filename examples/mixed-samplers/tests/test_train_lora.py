import argparse
import asyncio
import gzip
import json
import pickle
import signal
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from mixed_samplers import train_lora


def options(tmp_path, **overrides):
    return argparse.Namespace(
        **{
            "output": tmp_path,
            "lora_clients": 3,
            "fresh": False,
            "steps": 2,
            "inflight_prompts": 1,
            **overrides,
        }
    )


def service_mock(monkeypatch, *, fail_at=None, restore_fails=False):
    events = []
    clients = []

    def create(base_model, **kwargs):
        index = len(clients)
        events.append(("create", index))
        assert base_model == train_lora.config_lora.config.model
        assert kwargs == {
            "rank": 32,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": False,
        }
        if index == fail_at:
            raise RuntimeError("creation failed")
        future = Mock()
        future.result.return_value = SimpleNamespace(path=f"adapter-{index}")
        client = Mock(model_id=f"model-{index}")
        client.save_weights_for_sampler.return_value = future
        client.load_state_with_optimizer.return_value = future
        if restore_fails:
            client.load_state_with_optimizer.side_effect = RuntimeError(
                "restore failed"
            )
        clients.append(client)
        return client

    callbacks = []

    def register(*callback):
        events.append(("unload", len(callbacks)))
        callbacks.append(callback)

    monkeypatch.setattr(train_lora.atexit, "register", register)
    return (
        SimpleNamespace(create_lora_training_client=create),
        clients,
        events,
        callbacks,
    )


def test_sequential_creation_registers_unload_before_next_client(tmp_path, monkeypatch):
    service, clients, events, callbacks = service_mock(monkeypatch)
    runs = train_lora.create_clients(
        service,
        train_lora.config_lora.config.model,
        options(tmp_path),
        "https://spindle",
        "test-key",
    )
    assert events == [
        (kind, index) for index in range(3) for kind in ("create", "unload")
    ]
    for index, (training, args, seed) in enumerate(runs):
        assert training is clients[index]
        assert Path(args.output) == tmp_path / f"client-{index}"
        assert seed == train_lora.common.SEED + 1_000_003 * index
        assert args.start_step == 0 and args.checkpoint is None
        manifest = json.loads((Path(args.output) / "manifest.json").read_text())
        assert manifest["initial_publication"] == f"adapter-{index}"
        assert manifest["seed"] == seed
        assert callbacks[index] == (
            train_lora.unload_model,
            "https://spindle",
            "test-key",
            training.model_id,
        )


@pytest.mark.parametrize("restore_fails", [False, True])
def test_cleanup_registered_even_when_creation_or_restore_fails(
    tmp_path, monkeypatch, restore_fails
):
    service, _, events, callbacks = service_mock(
        monkeypatch, fail_at=1, restore_fails=restore_fails
    )
    if restore_fails:
        output = tmp_path / "client-0"
        output.mkdir()
        (output / "checkpoint.json").write_text(
            json.dumps({"step": 4, "path": "saved-state"})
        )
    with pytest.raises(
        RuntimeError, match="restore failed" if restore_fails else "creation failed"
    ):
        train_lora.create_clients(
            service,
            train_lora.config_lora.config.model,
            options(tmp_path),
            "https://spindle",
            "test-key",
        )
    assert events[:2] == [("create", 0), ("unload", 0)]
    assert len(callbacks) == 1


def test_resume_and_fresh_output_safety(tmp_path):
    args = options(tmp_path)
    output = tmp_path / "client-0"
    output.mkdir()
    (output / "checkpoint.json").write_text(json.dumps({"step": 4, "path": "state"}))
    resumed = train_lora.client_args(args, 0)
    assert (resumed.checkpoint, resumed.start_step) == ("state", 4)
    with pytest.raises(ValueError, match="requires an empty"):
        train_lora.client_args(options(tmp_path, fresh=True), 0)
    fresh = train_lora.client_args(options(tmp_path, fresh=True), 1)
    assert fresh.checkpoint is None and fresh.start_step == 0


def test_unload_endpoint_and_signals(monkeypatch, capsys):
    post = Mock()
    monkeypatch.setattr(train_lora.httpx, "post", post)
    train_lora.unload_model("https://spindle", "test-key", "model-1")
    post.assert_called_once_with(
        "https://spindle/api/v1/unload_model",
        headers={"X-API-Key": "test-key"},
        json={"model_id": "model-1"},
        timeout=300,
    )
    post.return_value.raise_for_status.assert_called_once()
    post.side_effect = httpx.ConnectError("offline")
    train_lora.unload_model("https://spindle", "test-key", "model-1")
    assert "trainer_cleanup_failed" in capsys.readouterr().out
    for signum in (signal.SIGINT, signal.SIGTERM):
        with pytest.raises(SystemExit) as exc:
            train_lora.exit_on_signal(signum, None)
        assert exc.value.code == 128 + signum


@pytest.mark.parametrize("fails", [False, True])
def test_clients_train_concurrently_on_shared_fleet(tmp_path, monkeypatch, fails):
    service, _, _, _ = service_mock(monkeypatch)
    clients = train_lora.create_clients(
        service,
        train_lora.config_lora.config.model,
        options(tmp_path),
        "https://spindle",
        "test-key",
    )
    gateways = {"b200": "url-0", "h200": "url-1"}
    tasks = [{"instance_id": "one"}]
    measured = []
    started = []
    stopped = []

    async def measure(url, model_id, version):
        measured.append((url, model_id, version))
        return {"tokens_per_second": 2}

    async def run(
        training, actual_gateways, weights, args, actual_tasks, *, seed, learning_rate
    ):
        assert actual_gateways is gateways and actual_tasks is tasks
        assert weights == [2, 2] and learning_rate == 1e-5
        assert seed == clients[int(training.model_id[-1])][2]
        started.append(training.model_id)
        try:
            while len(started) < 3:
                await asyncio.sleep(0)
            if fails and training.model_id == "model-2":
                raise RuntimeError("client failed")
            if fails:
                await asyncio.Event().wait()
        finally:
            stopped.append(training.model_id)

    monkeypatch.setattr(
        train_lora.measure_capacity, "remote", SimpleNamespace(aio=measure)
    )
    monkeypatch.setattr(train_lora.train, "train", run)
    if fails:
        with pytest.raises(ExceptionGroup, match="unhandled errors"):
            asyncio.run(
                asyncio.wait_for(
                    train_lora.run_clients(clients, gateways, tasks), timeout=1
                )
            )
    else:
        asyncio.run(
            asyncio.wait_for(
                train_lora.run_clients(clients, gateways, tasks), timeout=1
            )
        )
    assert sorted(started) == sorted(stopped) == ["model-0", "model-1", "model-2"]
    assert measured == [("url-0", "model-0", 1), ("url-1", "model-0", 1)]


def test_main_deploys_one_shared_lora_fleet_with_four_clients(tmp_path, monkeypatch):
    service, clients, _, _ = service_mock(monkeypatch)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(train_lora.tinker, "ServiceClient", Mock(return_value=service))
    monkeypatch.setattr(
        train_lora.modal.Function,
        "from_name",
        Mock(return_value=SimpleNamespace(get_web_url=lambda: "https://spindle")),
    )
    monkeypatch.setattr(train_lora.modal, "enable_output", nullcontext)
    monkeypatch.setattr(train_lora, "load_tasks", lambda: [{"instance_id": "one"}])
    monkeypatch.setattr(
        train_lora, "worker_app", SimpleNamespace(run=lambda **_: nullcontext())
    )
    signals = Mock()
    monkeypatch.setattr(train_lora.signal, "signal", signals)
    deployed = []

    def build(deployment, pool):
        assert isinstance(pool, train_lora.LoraPoolSpec)
        assert deployment.recipe.parameterization == "lora"
        deployed.append(deployment)
        return SimpleNamespace(run=lambda **_: nullcontext()), SimpleNamespace(
            get_url=lambda: str(len(deployed))
        )

    async def run(actual_clients, gateways, tasks):
        assert len(actual_clients) == 4 and len(gateways) == 8
        assert tasks == [{"instance_id": "one"}]

    monkeypatch.setattr(train_lora, "build_rollout_app", build)
    monkeypatch.setattr(train_lora, "run_clients", run)
    train_lora.main(["--output", str(tmp_path), "--fresh"])
    assert len(clients) == 4 and len(deployed) == 8
    assert signals.call_args_list[:2] == [
        ((signal.SIGINT, train_lora.exit_on_signal),),
        ((signal.SIGTERM, train_lora.exit_on_signal),),
    ]


@pytest.mark.parametrize("count", [0, -1, 7])
def test_invalid_client_count_rejected_before_allocating(tmp_path, count):
    with pytest.raises(SystemExit) as exc:
        train_lora.main(["--output", str(tmp_path), "--lora-clients", str(count)])
    assert exc.value.code == 2


def test_training_seed_and_learning_rate_override(tmp_path, monkeypatch):
    module = train_lora.train
    monkeypatch.setattr(module, "PROMPTS_PER_UPDATE", 1)
    monkeypatch.setattr(module, "TRAJECTORIES_PER_PROMPT", 2)
    seeds = []
    optimizers = []
    rng_seeds = []
    random_class = module.random.Random

    def rng(seed):
        rng_seeds.append(seed)
        return random_class(seed)

    monkeypatch.setattr(module.random, "Random", rng)

    async def sample(task, gateway, group, model_id, version, seed):
        seeds.append(seed)
        await asyncio.sleep(0)
        row = {
            "instance_id": task["instance_id"],
            "tokens": [1, 2],
            "active": [0, 1],
            "logq": [0.0, 0.0],
            "offsets": [0, 0, 1],
            "support": [2],
            "support_logq": [0.0],
            "reward": float(len(seeds) % 2),
            "format_error": False,
            "generated_tokens": 1,
            "versions": [{"start": version, "end": version}],
            "group": group,
            "started_at": 0,
            "finished_at": 1,
            "stop_reason": "submit",
            "turns": 1,
        }
        return gzip.compress(pickle.dumps(row))

    class Future:
        async def result_async(self):
            return SimpleNamespace(
                metrics={"update_successful:mean": 1.0}, path="publication"
            )

    training = SimpleNamespace(
        model_id="test",
        forward_backward=lambda *_: Future(),
        optim_step=lambda params: optimizers.append(params) or Future(),
        save_weights_for_sampler=lambda _: Future(),
        save_state=lambda _: Future(),
    )
    monkeypatch.setattr(
        module, "sample_trajectory", SimpleNamespace(remote=SimpleNamespace(aio=sample))
    )
    args = SimpleNamespace(
        output=str(tmp_path), inflight_prompts=1, start_step=0, steps=1
    )
    asyncio.run(
        asyncio.wait_for(
            module.train(
                training,
                {"b200-nvfp4": "url"},
                [1.0],
                args,
                [{"instance_id": "one"}],
                seed=123456,
                learning_rate=1e-5,
            ),
            timeout=2,
        )
    )
    assert rng_seeds == [123456]
    assert seeds[:2] == [123456, 124456]
    assert len(optimizers) == 1
    params = optimizers[0]
    assert (
        params.learning_rate,
        params.beta1,
        params.beta2,
        params.eps,
        params.grad_clip_norm,
    ) == (1e-5, 0.9, 0.95, 1e-8, 1.0)
