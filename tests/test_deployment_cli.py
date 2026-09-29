import json
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pytest

from spindle import deployment_cli as cli
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal.deployment_configs import (
    CONFIGS_ENV,
    PLATFORM_ENV,
    deployed_configs,
)
from spindle.providers.modal.fft_pool import FFTPoolSpec
from spindle.providers.modal.image_dependencies import ignore_config_source
from spindle.providers.modal.lora_pool import LoraPoolSpec


def deployment():
    return DeploymentConfig.create(load(config_path("qwen35-9b-lora-16k")))


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setattr(cli.sys, "version_info", (3, 12, 0))
    state = SimpleNamespace(
        apps=set(), configs=[], platform=None, pools=[], calls=[], fail=None
    )

    class FakeApp:
        def __init__(self, name, role):
            self.name = name
            self.role = role

        def deploy(self, environment_name=None):
            state.calls.append(self.role)
            if state.fail == self.role:
                raise RuntimeError(f"{self.role} deploy failed")
            state.apps.add(self.name)

    def read_configs(frontend, environment):
        return state.configs

    def lookup(name, **kwargs):
        if name not in state.apps:
            raise modal.exception.NotFoundError("not deployed")

    def run(command, *, env, check):
        state.calls.append("frontend")
        if state.fail == "frontend":
            raise subprocess.CalledProcessError(1, command)
        state.configs = json.loads(env[CONFIGS_ENV])
        state.platform = json.loads(env[PLATFORM_ENV])

    monkeypatch.setattr(cli, "deployed_configs", read_configs)
    monkeypatch.setattr(cli, "deployed_pools", lambda *args: state.pools)
    monkeypatch.setattr(cli, "deployed_platform", lambda *args: state.platform)
    monkeypatch.setattr(modal.App, "lookup", lookup)
    monkeypatch.setattr(
        cli,
        "build_trainer_app",
        lambda row, platform=None, **k: (
            FakeApp(row.trainer_app_name, "trainer"),
            None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_inference_app",
        lambda row, platform=None, **k: (
            FakeApp(row.inference_app_name, "inference"),
            None,
        ),
    )
    monkeypatch.setattr(cli.modal, "enable_output", lambda: nullcontext())
    monkeypatch.setattr(
        cli,
        "build_rollout_app",
        lambda row, pool, platform: (FakeApp(pool.app_name, "pool"), None),
    )
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        modal.Dict,
        "from_name",
        lambda *a, **k: pytest.fail("deployment must not use a Dict registry"),
    )
    return state


def test_deploy_defaults_to_bundled_configs(tmp_path, monkeypatch, deployed):
    bundled = config_path("qwen35-9b-lora-16k").parent.glob("[!_]*.py")
    expected = {load(path).name for path in bundled}
    monkeypatch.chdir(tmp_path)
    cli.main(["deploy"])
    assert len(expected) > 1
    assert {row["recipe"]["name"] for row in deployed.configs} == expected


def test_deploy_reuses_unchanged_apps(deployed):
    row = deployment()
    cli.deploy([row])
    assert deployed.calls == ["trainer", "inference", "frontend"]
    deployed.calls.clear()
    cli.deploy([row])
    assert deployed.calls == ["frontend"]

    spec = deepcopy(row.recipe)
    spec.inference_max_replicas = 6
    changed = DeploymentConfig.create(spec)
    deployed.calls.clear()
    cli.deploy([changed])
    assert deployed.calls == ["inference", "frontend"]

    assert [r["recipe"]["name"] for r in deployed.configs] == [changed.name]

    # Modal's actual state wins over a frontend config that mentions an old app.
    deployed.apps.remove(changed.inference_app_name)
    deployed.calls.clear()
    cli.deploy([changed])
    assert deployed.calls == ["inference", "frontend"]


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_inference_changes_refresh_existing_pools_and_retry_before_frontend(
    deployed, mode
):
    row = DeploymentConfig.create(
        load(
            config_path("qwen35-9b-lora-16k" if mode == "lora" else "qwen35-4b-fft-64k")
        )
    )
    cli.deploy([row])
    pool = (
        LoraPoolSpec(row.name)
        if mode == "lora"
        else FFTPoolSpec(row.name, "job", True, 0)
    )
    deployed.apps.add(pool.app_name)
    deployed.pools = [pool.as_dict(), {"definition_id": "unrelated"}]
    recipe = deepcopy(row.recipe)
    recipe.inference_max_replicas = 3
    changed = DeploymentConfig.create(recipe)
    deployed.calls.clear()
    deployed.fail = "pool"
    with pytest.raises(RuntimeError, match="pool deploy failed"):
        cli.deploy([changed])
    assert deployed.calls == ["inference", "pool"]
    assert deployed.configs[0]["recipe"]["inference_max_replicas"] == 8
    deployed.fail = None
    deployed.calls.clear()
    cli.deploy([changed])
    assert deployed.calls == ["inference", "pool", "frontend"]


@pytest.mark.parametrize(
    "failure,error",
    [("inference", RuntimeError), ("frontend", subprocess.CalledProcessError)],
)
def test_retry_discovers_completed_apps_without_pending_configs(
    deployed, failure, error
):
    row = deployment()
    deployed.fail = failure
    with pytest.raises(error):
        cli.deploy([row])
    assert deployed.configs == []
    assert row.trainer_app_name in deployed.apps
    deployed.calls.clear()
    deployed.fail = None
    cli.deploy([row])
    assert deployed.calls == (
        ["inference", "frontend"] if failure == "inference" else ["frontend"]
    )


def test_refresh_redeploys_only_that_trainer(deployed):
    miles = deployment()
    fft = DeploymentConfig.create(load(config_path("qwen35-4b-fft-64k")))
    cli.deploy([miles, fft])
    deployed.calls.clear()
    cli.deploy([miles, fft], refresh_trainers=[miles.name])
    assert deployed.calls == ["trainer", "frontend"]
    deployed.calls.clear()
    cli.deploy([miles, fft])
    assert deployed.calls == ["frontend"]


def test_region_change_updates_worker_placement(deployed):
    row = deployment()
    cli.deploy([row])
    platform = row.recipe.platform
    platform["modal"]["region"] = "us-east"
    deployed.calls.clear()
    cli.deploy([row])
    assert deployed.calls == ["trainer", "inference", "frontend"]
    assert deployed.platform["modal"]["region"] == "us-east"


def test_existing_frontend_without_deployment_metadata_can_be_updated(deployed):
    row = deployment()
    deployed.apps.add("spindle")
    cli.deploy([row])
    assert deployed.configs[0]["recipe"]["name"] == row.name


def test_read_configs_from_deployed_function(monkeypatch):
    function = SimpleNamespace(remote=Mock(return_value=[{"configuration": "saved"}]))
    lookup = Mock(return_value=function)
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    assert deployed_configs("my-app", "dev") == [{"configuration": "saved"}]
    lookup.assert_called_once_with("my-app", "deployed_configs", environment_name="dev")
    function.remote.side_effect = modal.exception.NotFoundError("no function")
    assert deployed_configs("my-app", "dev") == []
    function.remote.side_effect = RuntimeError("frontend failed")
    with pytest.raises(RuntimeError, match="frontend failed"):
        deployed_configs("my-app", "dev")


def test_validate_never_resolves_or_deploys(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "compile_configs",
        lambda *args, **kwargs: pytest.fail("unexpected compile"),
    )
    monkeypatch.setattr(
        cli, "deploy", lambda *args: pytest.fail("unexpected deployment")
    )
    cli.main(["config", "validate", str(config_path("qwen35-9b-lora-16k"))])
    assert "Validated 1 deployment" in capsys.readouterr().out


def test_deploy_rejects_python_mismatch_before_remote_changes(monkeypatch):
    monkeypatch.setattr(cli.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(
        modal.Dict,
        "from_name",
        lambda *a, **k: pytest.fail("must reject before touching Modal"),
    )
    with pytest.raises(ValueError, match="requires Python 3.12"):
        cli.deploy([deployment()])


def test_source_mount_excludes_authoring_configs():
    assert ignore_config_source(Path("configs/example.py"))
    assert ignore_config_source(Path("configs/__init__.py"))
    assert ignore_config_source(Path("data.json"))
    assert not ignore_config_source(Path("deployments.py"))
    assert not ignore_config_source(Path("backends/miles_config.py"))


def test_cli_flags_override_recipe_platform(monkeypatch):
    seen = {}

    def compile(paths):
        seen["compiled"] = paths
        return [deployment()]

    def deploy(configs, **kwargs):
        seen["configs"] = configs
        seen["platform"] = configs[0].recipe.platform
        seen.update(kwargs)

    monkeypatch.setattr(cli, "compile_configs", compile)
    monkeypatch.setattr(cli, "deploy", deploy)
    cli.main(
        [
            "deploy",
            "model.py",
            "--app",
            "my-spindle",
            "--env",
            "dev",
            "--region",
            "us-east",
            "--refresh-trainer",
            "my-model",
        ]
    )
    assert seen["platform"]["frontend"] == "my-spindle"
    assert seen["platform"]["modal"] == {"environment": "dev", "region": "us-east"}
    assert seen["refresh_trainers"] == ["my-model"]
    assert len(seen["configs"]) == 1


def test_builtin_config_keeps_the_model_path():
    (row,) = cli.compile_configs([config_path("qwen35-9b-lora-16k")])
    assert row.asset_path == "/assets/Qwen/Qwen3.5-9B-Base"
    assert not hasattr(load(config_path("qwen35-9b-lora-16k")), "revision")


def test_deploy_reads_platform_overrides_from_config_file(tmp_path, deployed):
    path = tmp_path / "model.py"
    path.write_text(
        "from spindle.configs.gpt_oss_20b_lora_64k import Config as Parent\n"
        "class Config(Parent):\n"
        "    overrides = {\n"
        '        "platform.frontend": "research",\n'
        '        "platform.modal.environment": "dev",\n'
        '        "platform.modal.region": "us-east",\n'
        '        "platform.secrets.api": "research-api",\n'
        '        "platform.storage.checkpoints": "research-checkpoints",\n'
        "    }\n"
        "config = Config()\n"
    )
    cli.main(["deploy", str(path)])
    platform = deployed.platform
    assert platform["frontend"] == "research"
    assert platform["modal"] == {"environment": "dev", "region": "us-east"}
    assert platform["secrets"]["api"] == "research-api"
    assert platform["secrets"]["sampler_proxy"] == "spindle-proxy"
    assert platform["storage"]["checkpoints"] == "research-checkpoints"
    assert deployed.configs[0]["recipe"]["platform"] == platform
    assert deployment().recipe.platform["frontend"] == "spindle"


def test_conflicting_platforms_fail_before_deployment(deployed):
    first = deployment()
    second = DeploymentConfig.create(load(config_path("qwen35-4b-fft-64k")))
    second.recipe.platform["secrets"]["api"] = "different-api"
    with pytest.raises(ValueError, match="share platform settings"):
        cli.deploy([first, second])
    assert deployed.calls == []
