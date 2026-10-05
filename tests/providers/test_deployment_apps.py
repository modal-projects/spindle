import asyncio
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import modal
import pytest
from modal._serialization import serialize

import spindle.backends.deployment as backend
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.local import InMemoryKeyValueStore
from spindle.providers.modal import (
    deployment_apps,
    deployment_configs,
    fft_pool,
    lora_pool,
)
from spindle.providers.modal.fft_pool import FFTPoolSpec
from spindle.providers.modal.lora_pool import LoraPoolSpec


def deployment(preset="qwen35-9b-lora-16k"):
    return DeploymentConfig.create(load(config_path(preset)))


class App:
    def __init__(self, name):
        self.name = name
        self.functions = {}
        self.servers = {}

    def function(self, **settings):
        def decorate(fn):
            self.functions[settings["name"]] = (settings, fn)
            return fn

        return decorate

    def server(self, **settings):
        def decorate(cls):
            self.servers[settings["name"]] = (settings, cls)
            return cls

        return decorate


@pytest.fixture
def builders(monkeypatch):
    monkeypatch.setattr(modal, "App", App)
    monkeypatch.setattr(modal, "enter", lambda: lambda fn: fn)
    monkeypatch.setattr(modal, "exit", lambda: lambda fn: fn)


@pytest.mark.parametrize(
    "preset,backend,clients,nproc",
    [
        ("qwen35-9b-lora-16k", "miles_lora", 6, 1),
        ("qwen35-4b-fft-64k", "megatron_fft", 1, 4),
    ],
)
def test_trainer_declaration_and_executor_configuration(
    builders, monkeypatch, preset, backend, clients, nproc
):
    row = deployment(preset)
    platform = row.recipe.platform
    platform["storage"]["checkpoints"] = "test-custom-checkpoints"
    image = object()
    app, trainer = deployment_apps.build_trainer_app(row, platform, image=image)
    declaration, _ = app.functions["trainer"]
    assert declaration["gpu"] == "H100:4"
    assert declaration["region"] == "us-west"
    assert declaration["max_containers"] is None
    assert declaration["single_use_containers"] is True
    assert declaration["image"] is image
    calls = []
    reloaded = []

    monkeypatch.setattr(deployment_apps, "shared_kv", lambda: "store")
    monkeypatch.setattr(
        deployment_apps,
        "volumes_for",
        lambda spec: {"/assets": SimpleNamespace(reload=lambda: reloaded.append(True))},
    )
    monkeypatch.setattr(
        deployment_apps,
        "run_engine_with_backend",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setenv("SPINDLE_FRONTEND_APP_ID", "previous")
    trainer("instance-a", row.model_dump_json(), "ap-frontend")
    assert os.environ["SPINDLE_FRONTEND_APP_ID"] == "ap-frontend"
    args, kwargs = calls[0]
    assert args == ("store", f"spindle.backends.{backend}:build_executor")
    assert kwargs["max_models"] == clients
    assert kwargs["nproc"] == nproc
    assert (
        kwargs["backend_env"]["SPINDLE_CHECKPOINT_VOLUME"] == "test-custom-checkpoints"
    )
    assert kwargs["backend_env"]["SPINDLE_BASE_MODEL"] == row.model
    config = json.loads(kwargs["backend_env"]["SPINDLE_BACKEND_CONFIG"])
    assert config[row.recipe.backend]["hf_checkpoint"] == row.asset_path
    assert config["checkpoint_dir"] == "/checkpoints"
    assert reloaded == [True]


@pytest.mark.parametrize("kind", ["lora", "base", "latest", "pinned"])
def test_pool_starts_native_server_and_correct_sidecar(builders, monkeypatch, kind):
    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    pool = (
        LoraPoolSpec(row.definition_id)
        if kind == "lora"
        else (
            FFTPoolSpec.base(row.definition_id)
            if kind == "base"
            else FFTPoolSpec(row.definition_id, "job", kind == "latest", 7)
        )
    )
    app, server = deployment_apps.build_rollout_app(row, pool, image="test-image")
    settings, _ = app.servers["Server"]
    assert app.name == pool.app_name
    assert (
        settings["gpu"]
        == f"{row.recipe.inference_gpu}:{row.recipe.inference_gpus_per_node}"
    )
    assert settings["min_containers"] == 0
    assert settings["target_concurrency"] == 16
    assert settings["compute_region"] == "us-west"

    calls, commands, stops = [], [], []
    process = object()
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kw: commands.append(argv) or process
    )
    monkeypatch.setattr(deployment_apps, "wait_http", lambda *args: None)
    monkeypatch.setattr(deployment_apps, "supervise_children", lambda *args: None)
    monkeypatch.setattr(
        deployment_apps,
        "start_lora_sidecar",
        lambda **kw: calls.append(("lora", kw)) or process,
    )
    monkeypatch.setattr(
        deployment_apps,
        "start_fft_sidecar",
        lambda **kw: calls.append(("fft", kw)) or process,
    )
    monkeypatch.setattr(deployment_apps, "terminate", stops.append)
    replica = server()
    replica.start()
    assert commands[0][2] == "spindle.inference.sglang"
    assert commands[0][3] == row.asset_path
    native = json.loads(commands[0][4])
    assert native["context_length"] == row.max_context_length
    if kind == "lora":
        assert native["enable_lora"] is True
        assert native["max_lora_rank"] == 32
        assert native["lora_target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    else:
        assert native["enable_cpu_weight_cache"] is True
        assert calls[0][1]["pinned_version"] == (
            None if kind == "latest" else 0 if kind == "base" else 7
        )
    replica.stop()
    assert stops == [process, process]


def test_pool_lookup_uses_definition_name(monkeypatch):
    row = deployment()
    monkeypatch.setenv(deployment_configs.CONFIGS_ENV, json.dumps([row.model_dump()]))
    saved = deployment_configs.pool_config(row.definition_id)
    assert saved.definition_id == row.definition_id
    assert deployment_configs.pool_config("missing") is None


def test_startup_failure_is_visible_and_blocks_new_spawns(monkeypatch):
    app = importlib.import_module("spindle.providers.modal.app")
    store = InMemoryKeyValueStore()
    row = deployment()
    monkeypatch.setattr(app, "shared_kv", lambda: store)

    async def run():
        await store.put(
            f"deployment_failure:{row.definition_id}",
            {
                "error": "backend exited with code 1",
                "instance_id": "failed-instance",
                "failed_at": app.time.time(),
            },
        )
        with pytest.raises(ValueError, match="Trainer startup failed.*failed-instance"):
            await app._spawn_engine(row.definition_id, "new-instance")
        assert await app.deployment_error("legacy") is None

    asyncio.run(run())


def test_real_modal_app_constructs_from_configs_without_legacy_catalog(monkeypatch):
    row = deployment()
    env = {
        **os.environ,
        deployment_configs.CONFIGS_ENV: json.dumps([row.model_dump()]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib, sys, modal
from spindle.providers.modal import deployment_apps, deployment_configs
deployment_apps.image_for = lambda backend: modal.Image.debian_slim()
app = importlib.import_module('spindle.providers.modal.app')
assert len(app.DEFINITIONS) == 1
assert app.APP_NAME == 'spindle'
assert app.deployed_configs.local() == [d.model_dump(mode='json') for d in app.configs_from_env()]
assert app.DEFINITIONS[0].trainer_app_name
assert not any(name.startswith('spindle.providers.modal.definitions.') for name in sys.modules)
print('constructed')
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "constructed" in result.stdout


def test_admission_changes_preserve_serialized_trainer(builders):
    first = deployment()
    old_bytes = serialize(deployment_apps.build_trainer_app(first, image="test")[1])
    changed = first.model_copy(deep=True)
    changed.recipe.trainer_timeout_s = 1
    new_bytes = serialize(deployment_apps.build_trainer_app(changed, image="test")[1])
    assert new_bytes == old_bytes
    changed.recipe.trainer_gpu = "H200"
    assert (
        serialize(deployment_apps.build_trainer_app(changed, image="test")[1])
        == old_bytes
    )
    changed.recipe.name = "other-model"
    assert (
        serialize(deployment_apps.build_trainer_app(changed, image="test")[1])
        != old_bytes
    )


@pytest.mark.parametrize("kind", ["lora", "full"])
def test_pool_launch_uses_only_generic_deployment_app(monkeypatch, kind):
    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    monkeypatch.setenv(deployment_configs.CONFIGS_ENV, json.dumps([row.model_dump()]))
    module = lora_pool if kind == "lora" else fft_pool
    spec = (
        LoraPoolSpec(row.definition_id)
        if kind == "lora"
        else FFTPoolSpec(row.definition_id, "model", True, 0)
    )
    calls = []

    class Pool:
        def __init__(self, *args):
            self.lookups = 0

        def gateway_url(self):
            self.lookups += 1
            if self.lookups == 1:
                raise modal.exception.NotFoundError("not deployed")
            return "https://pool"

    monkeypatch.setattr(module, "ModalFlashPool", Pool)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/bin/modal")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    assert module.deploy_pool(spec, config=row) == "https://pool"
    command, kwargs = calls[0]
    assert (
        command[command.index("-m") + 1]
        == "spindle.providers.modal.deployment_pool_app"
    )
    assert (
        json.loads(kwargs["env"][deployment_configs.POOL_CONFIG_ENV])["recipe"]["name"]
        == row.name
    )


@pytest.mark.parametrize("kind", ["lora", "full"])
def test_missing_pool_uses_saved_provisioner(monkeypatch, kind):
    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    monkeypatch.setenv(deployment_configs.CONFIGS_ENV, json.dumps([row.model_dump()]))
    module = lora_pool if kind == "lora" else fft_pool
    spec = (
        LoraPoolSpec(row.definition_id)
        if kind == "lora"
        else FFTPoolSpec.base(row.definition_id)
    )

    class MissingPool:
        def __init__(self, *args):
            pass

        def gateway_url(self):
            raise modal.exception.NotFoundError("not deployed")

    monkeypatch.setattr(module, "ModalFlashPool", MissingPool)
    calls = []
    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda name, function, **kw: (
            calls.append((name, function))
            or SimpleNamespace(remote=lambda record, pool: "https://saved-runtime")
        ),
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("frontend rebuilt pool")
    )
    assert module.deploy_pool(spec) == "https://saved-runtime"
    assert calls == [(row.inference_app_name, "provision")]


def test_provisioner_rejects_wrong_settings_and_uses_saved_config(
    builders, monkeypatch
):
    row = deployment()
    app, provision = deployment_apps.build_inference_app(row, image="test")
    assert app.name == row.inference_app_name
    calls = []
    monkeypatch.setattr(
        deployment_apps,
        "deploy_lora",
        lambda pool, *, config, platform=None: (
            calls.append((pool, config)) or "https://pool"
        ),
    )
    pool = LoraPoolSpec(row.definition_id)
    assert provision(row.model_dump_json(), pool.as_dict()) == "https://pool"
    assert calls[0][1].name == row.name
    changed = row.model_copy(deep=True)
    changed.recipe.name = "other-model"
    with pytest.raises(ValueError, match="inference settings"):
        provision(changed.model_dump_json(), pool.as_dict())


@pytest.mark.parametrize("role", ["trainer", "inference"])
def test_trainer_and_inference_apps_construct_offline(builders, role):
    row = deployment()
    if role == "trainer":
        app, _ = deployment_apps.build_trainer_app(row, image="test")
    else:
        app, _ = deployment_apps.build_inference_app(row, image="test")
    assert app.name == (
        row.trainer_app_name if role == "trainer" else row.inference_app_name
    )


def test_spawn_passes_job_configuration_to_saved_trainer(monkeypatch):
    app = importlib.import_module("spindle.providers.modal.app")
    row = deployment()
    calls = []

    async def spawn(instance_id, config_json, frontend_app_id):
        calls.append((instance_id, config_json, frontend_app_id))
        return SimpleNamespace(object_id="call-id")

    async def no_error(definition_id):
        return None

    monkeypatch.setattr(app, "deployment_error", no_error)
    monkeypatch.setattr(app, "current_app_id", lambda: "ap-frontend")
    monkeypatch.setattr(app, "module_for", lambda _: row)
    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda name, function, **kwargs: SimpleNamespace(
            spawn=SimpleNamespace(aio=spawn)
        ),
    )
    assert asyncio.run(app._spawn_engine(row.definition_id, "instance")) == "call-id"
    assert calls == [("instance", row.model_dump_json(), "ap-frontend")]


def test_declared_compute_settings_reach_modal(builders):
    spec = deepcopy(deployment().recipe)
    spec.trainer_timeout_s = 90
    spec.trainer_cpu = 12
    spec.trainer_memory_mib = 123456
    spec.inference_startup_timeout_s = 90
    spec.inference_min_replicas = 1
    spec.inference_max_replicas = 3
    spec.inference_cpu = 6
    spec.inference_memory_mib = 45000
    row = DeploymentConfig.create(spec)
    trainer_app, _ = deployment_apps.build_trainer_app(row, image="test")
    trainer, _ = trainer_app.functions["trainer"]
    assert (trainer["cpu"], trainer["memory"], trainer["timeout"]) == (12, 123456, 90)
    pool_app, _ = deployment_apps.build_rollout_app(
        row, LoraPoolSpec(row.definition_id), image="test"
    )
    server, _ = pool_app.servers["Server"]
    assert (server["cpu"], server["memory"], server["startup_timeout"]) == (
        6,
        45000,
        90,
    )
    assert (server["min_containers"], server["max_containers"]) == (1, 3)


def test_multinode_trainer_uses_cluster_launcher(builders, monkeypatch):
    row = deployment("qwen38-27b-lora-256k")
    clusters = []

    def clustered(nodes, *, rdma):
        clusters.append((nodes, rdma))

        def decorate(fn):
            return fn

        return decorate

    monkeypatch.setattr(modal.experimental, "clustered", clustered)
    app, _ = deployment_apps.build_trainer_app(row, image="test")
    settings, _ = app.functions["trainer"]
    assert clusters == [(2, True)]
    assert settings["gpu"] == "H200:8"
    assert settings["experimental_options"] == {"efa_enabled": True}
    calls = []
    monkeypatch.setattr(deployment_apps, "shared_kv", lambda: "store")
    monkeypatch.setattr(
        deployment_apps,
        "volumes_for",
        lambda _: {"/assets": SimpleNamespace(reload=lambda: None)},
    )
    monkeypatch.setattr(
        deployment_apps,
        "start_trainer_cluster",
        lambda nodes, **kwargs: "10.0.0.1:6379",
    )
    monkeypatch.setattr(
        deployment_apps, "run_engine_with_backend", lambda *a, **kw: calls.append(kw)
    )
    deployment_apps.run_trainer(row, "instance")
    assert calls[0]["backend_env"]["SPINDLE_RAY_ADDRESS"] == "10.0.0.1:6379"
    assert (
        json.loads(calls[0]["backend_env"]["SPINDLE_BACKEND_CONFIG"])["miles"][
            "actor_num_nodes"
        ]
        == 2
    )
    monkeypatch.setattr(deployment_apps, "start_trainer_cluster", lambda *a, **k: None)
    deployment_apps.run_trainer(row, "worker")
    assert len(calls) == 1


def test_launchers_do_not_reparse_backend_config(builders, monkeypatch):
    row = deployment()

    def unexpected(*args, **kwargs):
        pytest.fail("launcher must use the saved resolved settings")

    monkeypatch.setattr(backend, "backend_config", unexpected)
    monkeypatch.setattr(backend, "serving_options", unexpected)
    deployment_apps.build_trainer_app(row, image="test")
    deployment_apps.build_rollout_app(
        row, LoraPoolSpec(row.definition_id), image="test"
    )


def test_compute_region_can_differ_from_http_routing(builders):
    row = deployment()
    row.recipe.platform["modal"].update(region="us-east", compute_region="us")
    trainer_app, _ = deployment_apps.build_trainer_app(row, image="test")
    trainer, _ = trainer_app.functions["trainer"]
    pool_app, _ = deployment_apps.build_rollout_app(
        row, LoraPoolSpec(row.definition_id), image="test"
    )
    server, _ = pool_app.servers["Server"]
    assert trainer["region"] == "us"
    assert server["compute_region"] == "us"
    assert server["routing_region"] == "us-east"
