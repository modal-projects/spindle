import asyncio
import json
import subprocess
import threading
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import modal
import pytest
from rich.console import Console

from spindle import deploy_wizard
from spindle import deployment_cli as cli
from spindle.control_plane.deployments import REGISTRY_VERSION_KEY
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.local import InMemoryKeyValueStore
from spindle.providers.modal import deployment_configs
from spindle.providers.modal.deployment_configs import (
    FRONTEND_SETTINGS_ENV,
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
        apps=set(),
        configs=[],
        platform=None,
        pools=[],
        calls=[],
        fail=None,
        images=[],
        weights=[],
    )

    class FakeApp:
        def __init__(self, name, role):
            self.name = name
            self.role = role

        def deploy(self, environment_name=None, tag=""):
            state.calls.append(self.role)
            if state.fail == self.role:
                raise RuntimeError(f"{self.role} deploy failed")
            state.apps.add(self.name)

    def read_configs(frontend, environment):
        return state.configs

    def lookup(name, **kwargs):
        if name not in state.apps:
            raise modal.exception.NotFoundError("not deployed")

    def run(command, *, env, check, **kwargs):
        state.calls.append("frontend")
        if state.fail == "frontend":
            raise subprocess.CalledProcessError(1, command)
        state.platform = json.loads(env[FRONTEND_SETTINGS_ENV])["platform"]
        state.env = env

    def write_registry(frontend, environment, configs, *, keep_existing=()):
        if state.platform is None:
            return False  # the frontend is not deployed yet
        registry = {row["recipe"]["name"]: row for row in state.configs}
        for row in keep_existing:
            registry.setdefault(row["recipe"]["name"], row)
        for row in configs:
            registry[row["recipe"]["name"]] = row
        state.configs = list(registry.values())
        return True

    monkeypatch.setattr(cli, "deployed_configs", read_configs)
    monkeypatch.setattr(cli, "write_registry", write_registry)
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
        cli,
        "build_sampler_image",
        lambda frontend, environment: state.images.append((frontend, environment)),
    )

    def start_weights(frontend, environment, definition_id):
        state.weights.append((definition_id, "frontend" in state.calls))
        return True

    monkeypatch.setattr(cli, "start_weights_download", start_weights)
    monkeypatch.setattr(
        modal.Dict,
        "from_name",
        lambda *a, **k: pytest.fail("deployment must not use a Dict registry"),
    )
    return state


def test_deploy_all_deploys_bundled_configs(tmp_path, monkeypatch, deployed):
    bundled = config_path("qwen35-9b-lora-16k").parent.glob("[!_]*.py")
    expected = {load(path).name for path in bundled}
    monkeypatch.chdir(tmp_path)
    cli.main(["deploy", "--all"])
    assert len(expected) > 1
    assert {row["recipe"]["name"] for row in deployed.configs} == expected
    assert max(len(value) for value in deployed.env.values()) < 131_072


def registry(monkeypatch, rows, app_id="ap-1"):
    """Point deployed_configs at an in-memory registry folder."""
    store = InMemoryKeyValueStore({row["recipe"]["name"]: row for row in rows})
    monkeypatch.setattr(deployment_configs, "frontend_app_id", lambda *a: app_id)
    monkeypatch.setattr(deployment_configs, "deployment_kv", lambda *a: store)


def test_unresponsive_frontend_fails_fast(monkeypatch):
    registry(monkeypatch, [])
    cancelled = []

    class Call:
        def get(self, timeout):
            raise TimeoutError

        def cancel(self):
            cancelled.append(True)

    class Function:
        def spawn(self):
            return Call()

    monkeypatch.setattr(modal.Function, "from_name", lambda *a, **k: Function())
    with pytest.raises(ValueError, match="modal app stop spindle"):
        deployment_configs.deployed_configs("spindle", "dev")
    assert cancelled == [True]


def test_deploy_accepts_bundled_config_names(deployed):
    cli.main(["deploy", "qwen35-9b-lora-16k", "qwen3-8b-lora-32k"])
    assert [row["recipe"]["name"] for row in deployed.configs] == [
        "qwen35-9b-lora-16k",
        "qwen3-8b-lora-32k",
    ]


def test_deploy_rejects_unknown_config_names(deployed, capsys):
    with pytest.raises(SystemExit):
        cli.main(["deploy", "not-a-model"])
    assert "unknown config 'not-a-model'" in capsys.readouterr().err
    assert deployed.calls == []


def test_bare_deploy_without_a_terminal_deploys_nothing(monkeypatch, deployed, capsys):
    monkeypatch.setattr(cli, "interactive", lambda: False)
    with pytest.raises(SystemExit):
        cli.main(["deploy"])
    assert "--all" in capsys.readouterr().err
    assert deployed.calls == []


def test_bare_deploy_in_a_terminal_deploys_the_picked_models(monkeypatch, deployed):
    presets = deploy_wizard.bundled_presets()
    picked = [deploy_wizard.pick(presets, "Qwen/Qwen3.5-9B-Base", "lora", 32_768)]
    shown = []
    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr(deploy_wizard, "choose", lambda *args: picked)
    monkeypatch.setattr(
        deploy_wizard,
        "show_next_steps",
        lambda console, chosen, *args: shown.append(chosen),
    )
    cli.main(["deploy"])
    assert [row["recipe"]["name"] for row in deployed.configs] == ["qwen35-9b-lora-64k"]
    assert deployed.calls == ["trainer", "inference", "frontend"]
    assert shown == [picked]


def test_progress_run_names_every_app_and_the_retry_command(monkeypatch, deployed):
    console = Console(record=True, width=120)
    presets = deploy_wizard.bundled_presets()
    picked = [deploy_wizard.pick(presets, "Qwen/Qwen3.5-9B-Base", "lora", 16_384)]
    deployed.fail = "inference"
    with pytest.raises(RuntimeError):
        cli.deploy(
            cli.compile_configs([preset.path for preset in picked]),
            run_steps=partial(
                deploy_wizard.deploy_with_progress,
                console,
                retry_command=deploy_wizard.deploy_command(picked),
            ),
        )
    output = console.export_text()
    assert "Trainer    qwen35-9b-lora-16k" in output
    assert "Inference  qwen35-9b-lora-16k failed" in output
    assert "spindle deploy qwen35-9b-lora-16k" in output
    assert deployed.calls == ["trainer", "inference"]


def test_sampler_image_failure_still_deploys_the_frontend(
    monkeypatch, deployed, capsys
):
    def broken(frontend, environment):
        raise RuntimeError("image build failed")

    monkeypatch.setattr(cli, "build_sampler_image", broken)
    cli.deploy([deployment()])
    assert deployed.calls == ["trainer", "inference", "frontend"]
    assert "Sampler image build failed: image build failed" in capsys.readouterr().err


def test_frontend_deploys_while_the_sampler_image_builds(monkeypatch, deployed):
    frontend_deployed = threading.Event()
    deploy_frontend = subprocess.run

    def build(frontend, environment):
        assert frontend_deployed.wait(timeout=5)
        deployed.images.append(frontend)

    def run(*args, **kwargs):
        deploy_frontend(*args, **kwargs)
        frontend_deployed.set()

    monkeypatch.setattr(cli, "build_sampler_image", build)
    monkeypatch.setattr(subprocess, "run", run)
    cli.deploy([deployment()])
    assert deployed.calls == ["trainer", "inference", "frontend"]
    assert deployed.images == ["spindle"]


def test_deploy_starts_weight_downloads_on_the_new_frontend(deployed, capsys):
    cli.deploy([deployment()])
    assert deployed.weights == [(deployment().definition_id, True)]
    assert (
        "Weights qwen35-9b-lora-16k: downloading in the background"
        in capsys.readouterr().out
    )


def test_cancelled_wizard_deploys_nothing(monkeypatch, deployed, capsys):
    def cancel(*args):
        raise deploy_wizard.Cancelled

    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr(deploy_wizard, "choose", cancel)
    with pytest.raises(SystemExit):
        cli.main(["deploy"])
    assert "cancelled" in capsys.readouterr().err
    assert deployed.calls == []


def test_new_spindle_version_redeploys_trainer_and_inference(deployed):
    row = deployment()
    cli.deploy([row])
    deployed.configs = [
        {**config, "spindle_version": "0.1.0"} for config in deployed.configs
    ]
    deployed.calls.clear()
    cli.deploy([row])
    assert deployed.calls == ["trainer", "inference", "frontend"]


def test_configs_saved_before_versioning_report_0_1_0():
    saved = deployment().model_dump(mode="json")
    del saved["spindle_version"]
    assert DeploymentConfig.model_validate(saved).spindle_version == "0.1.0"


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


def test_read_configs_from_the_registry_without_calling_the_frontend(monkeypatch):
    row = {"recipe": {"name": "qwen"}}
    registry(monkeypatch, [row])
    lookup = Mock(side_effect=AssertionError("the frontend was called"))
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    assert deployed_configs("my-app", "dev") == [row]
    registry(monkeypatch, [row], app_id=None)
    assert deployed_configs("my-app", "dev") == []


def test_frontend_from_before_the_registry_reports_its_configs(monkeypatch):
    registry(monkeypatch, [])
    call = SimpleNamespace(get=Mock(return_value=[{"configuration": "saved"}]))
    function = SimpleNamespace(spawn=Mock(return_value=call))
    lookup = Mock(return_value=function)
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    assert deployed_configs("my-app", "dev") == [{"configuration": "saved"}]
    lookup.assert_called_once_with("my-app", "deployed_configs", environment_name="dev")
    function.spawn.side_effect = modal.exception.NotFoundError("no function")
    assert deployed_configs("my-app", "dev") == []
    function.spawn.side_effect = None
    call.get.side_effect = RuntimeError("frontend failed")
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
        "from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent\n"
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


def served_elsewhere(name="unlisted-but-served"):
    row = DeploymentConfig.create(load(config_path("qwen35-4b-fft-64k")))
    data = row.model_dump(mode="json")
    data["recipe"]["name"] = name
    return data


def test_deploy_adds_listed_configs_and_keeps_unlisted_ones(deployed):
    cli.deploy([deployment()])
    other = served_elsewhere()
    deployed.configs.append(other)
    deployed.calls.clear()
    cli.deploy([deployment()])
    assert other in deployed.configs
    assert deployed.calls == ["frontend"]


def test_deploy_carries_over_a_pre_registry_frontends_recipes(deployed):
    other = served_elsewhere()
    deployed.configs = [other]
    deployed.platform = deployment().recipe.platform
    cli.deploy([deployment()])
    assert {row["recipe"]["name"] for row in deployed.configs} == {
        "unlisted-but-served",
        deployment().name,
    }


def test_platform_change_refuses_to_strand_unlisted_recipes(deployed):
    deployed.configs = [served_elsewhere()]
    deployed.platform = {**deployment().recipe.platform, "url_label": "old"}
    with pytest.raises(ValueError, match="unlisted-but-served"):
        cli.deploy([deployment()])
    assert deployed.calls == []


def test_lifecycle_change_refuses_to_reset_unlisted_recipes(deployed):
    other = served_elsewhere()
    other["recipe"]["session_idle_timeout_s"] = 86_400
    deployed.configs = [other]
    deployed.platform = deployment().recipe.platform
    with pytest.raises(
        ValueError, match=r"session_idle_timeout_s for \['unlisted-but-served'\]"
    ):
        cli.deploy([deployment()])
    assert deployed.calls == []


def mock_frontend_client(monkeypatch, handler):
    client_type = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kwargs: client_type(transport=transport, **kwargs),
    )
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example/")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")


def frontend_that_deploys(statuses, requests=None):
    """A fake frontend: POST registers, then the listing reports each status in turn."""
    statuses = iter(statuses)

    def handle(request):
        if requests is not None:
            requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"deployment": {}, "config": {}})
        return httpx.Response(
            200, json={"deployments": [{"name": "variant", **next(statuses)}]}
        )

    return handle


def test_add_posts_json_parsed_fields_then_waits_until_the_apps_are_ready(
    monkeypatch, deployed, capsys
):
    requests = []
    handle = frontend_that_deploys(
        [{"apps": "deploying"}, {"apps": "deploying"}, {"apps": "ready"}], requests
    )
    mock_frontend_client(monkeypatch, handle)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    cli.main(
        [
            "add",
            "variant",
            "--base",
            "qwen35-9b-lora-16k",
            "--trainer_gpus_per_node=4",
            "--weights_revision",
            '"1234"',
            "--trainer_gpu=H200",
            "--inference_gpu",
            '["H100","B300"]',
            "--model_weights=null",
        ]
    )
    request = requests[0]
    assert request.url == "https://frontend.example/api/v1/spindle/deployments"
    assert request.headers["x-api-key"] == "test-key"
    assert json.loads(request.content) == {
        "name": "variant",
        "base": "qwen35-9b-lora-16k",
        "overrides": {
            "trainer_gpus_per_node": 4,
            "weights_revision": "1234",
            "trainer_gpu": "H200",
            "inference_gpu": ["H100", "B300"],
            "model_weights": None,
        },
    }
    assert [r.method for r in requests] == ["POST", "GET", "GET", "GET"]
    assert deployed.calls == []
    assert "Added variant." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--not_a_setting=1"], "unknown recipe field"),
        (["--trainer_gpu=H100", "--trainer_gpu=B200"], "duplicate field"),
        (["--trainer_gpu"], "missing value"),
        (["--trainer_gpu", "--trainer_gpus_per_node=4"], "missing value"),
        (["--=1"], "expected a --FIELD=VALUE recipe field"),
        (["extra"], "expected a --FIELD=VALUE recipe field"),
        (["--all"], "unknown recipe field: all"),
        (["--env", "dev"], "unknown recipe field: env"),
        (["--set", "gpu=H200"], "unknown recipe field: set"),
        (["--name=other"], "not --name"),
    ],
)
def test_add_cli_validation_fails_before_http(extra_args, message, monkeypatch, capsys):
    requests = []
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: requests.append(kwargs))
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        cli.main(["add", "variant", "--base", "base", *extra_args])
    assert exc.value.code == 1
    assert message in capsys.readouterr().err
    assert requests == []


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["add", "variant"], 2, "--base"),
        (["add", "--base", "base"], 2, "name"),
        (["add", "variant", "--base"], 2, "expected one argument"),
        (["add", "variant", "--bas", "base"], 2, "--base"),
        (["deploy", "--trainer_gpu=H200"], 2, "unrecognized arguments"),
        (["remove", "x", "--all"], 2, "unrecognized arguments"),
        (["add", "variant", "--base", "base"], 1, "TINKER_BASE_URL"),
    ],
)
def test_add_cli_rejects_bad_invocations_before_http(
    args, code, message, monkeypatch, capsys
):
    requests = []
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: requests.append(kwargs))
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == code
    assert message in capsys.readouterr().err
    assert requests == []


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (
            400,
            {"error": "invalid_request", "message": "unknown base deployment"},
            "unknown base deployment",
        ),
        (409, {"error": "conflict", "message": "different settings"}, "409"),
        (405, {"message": "method not allowed"}, "predates the deployments API"),
    ],
)
def test_add_http_errors_exit(status, body, message, monkeypatch, capsys):
    mock_frontend_client(monkeypatch, lambda request: httpx.Response(status, json=body))
    with pytest.raises(SystemExit) as exc:
        cli.main(["add", "variant", "--base", "base"])
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def test_remove_deletes_the_named_config_with_the_proxy_token_when_set(
    monkeypatch, capsys
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"deployment": {"name": "variant"}})

    mock_frontend_client(monkeypatch, handle)
    cli.main(["remove", "variant"])
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    cli.remove_deployment("variant")
    assert [(r.method, r.url.path) for r in requests] == [
        ("DELETE", "/api/v1/spindle/deployments/variant")
    ] * 2
    assert requests[0].headers["X-API-Key"] == "test-key"
    assert "modal-key" not in requests[0].headers
    assert requests[1].headers["modal-key"] == "wk-test"
    assert requests[1].headers["modal-secret"] == "ws-test"
    assert "Removed variant" in capsys.readouterr().out


def test_remove_reports_unknown_names_and_old_frontends(monkeypatch):
    responses = iter(
        [
            httpx.Response(
                404, json={"error": "not_found", "message": "no deployment named 'x'"}
            ),
            httpx.Response(405),
        ]
    )
    mock_frontend_client(monkeypatch, lambda request: next(responses))
    with pytest.raises(ValueError, match="no deployment named 'x'"):
        cli.remove_deployment("x")
    with pytest.raises(ValueError, match="predates the deployments API"):
        cli.remove_deployment("x")


def test_app_flag_names_every_app_after_the_frontend(deployed):
    cli.main(["deploy", "qwen35-9b-lora-16k", "--app", "my-spindle", "--env", "dev"])
    assert deployed.apps == {
        "my-spindle-trainer-qwen35-9b-lora-16k",
        "my-spindle-inference-qwen35-9b-lora-16k",
    }
    assert deployed.env["SPINDLE_APP_NAME"] == "my-spindle"
    assert deployed.images == [("my-spindle", "dev")]
    # A pool from a release that named every pool spindle-... is not refreshed.
    old = LoraPoolSpec("qwen35-9b-lora-16k")
    deployed.apps.add(old.app_name)
    deployed.pools = [old.as_dict()]
    deployed.calls.clear()
    cli.main(["deploy", "qwen35-9b-lora-16k", "--app", "my-spindle", "--env", "dev"])
    assert deployed.calls == ["frontend"]


def test_stop_stops_the_frontend_then_pool_creators_then_pools(monkeypatch, capsys):
    listed = [
        ("my-spindle-lora-qwen35-9b-lora-16k", "deployed"),
        ("my-spindle-trainer-qwen35-9b-lora-16k", "deployed"),
        ("my-spindle-trainer-qwen3-8b-lora-32k", "stopped"),
        ("my-spindle-trainer-removed", "stopping..."),
        ("my-spindle-inference-qwen35-9b-lora-16k", "deployed"),
        ("my-spindle-fft-0123456789abcdef-latest", "deployed"),
        ("my-spindle", "deployed"),
        ("my-spindle-2", "deployed"),
        ("spindle-trainer-qwen35-9b-lora-16k", "deployed"),
        (None, "ephemeral"),
    ]
    calls = []

    def run(command, **kwargs):
        calls.append(command[3:])
        if command[3:5] == ["app", "list"]:
            rows = [{"description": name, "state": state} for name, state in listed]
            return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")
        if command[5].endswith("-latest"):
            return subprocess.CompletedProcess(
                command, 1, "", "App is already stopped."
            )
        if command[5].startswith("my-spindle-inference"):
            # Stopped by a DELETE after the listing was taken.
            return subprocess.CompletedProcess(
                command, 1, "", "No App with name 'x' found in the 'dev' environment."
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    deleted = []
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(cli, "frontend_app_id", lambda frontend, env: f"ap-{frontend}")
    monkeypatch.setattr(
        cli, "delete_registry", lambda app_id, env: deleted.append((app_id, env))
    )
    cli.main(["stop", "--app", "my-spindle", "--env", "dev"])
    assert deleted == [("ap-my-spindle", "dev")]
    assert [command[3] for command in calls[1:]] == [
        "my-spindle",
        "my-spindle-inference-qwen35-9b-lora-16k",
        "my-spindle-fft-0123456789abcdef-latest",
        "my-spindle-lora-qwen35-9b-lora-16k",
        "my-spindle-trainer-qwen35-9b-lora-16k",
    ]
    assert calls[:2] == [
        ["app", "list", "--json", "--env", "dev"],
        ["app", "stop", "-y", "my-spindle", "--env", "dev"],
    ]
    assert capsys.readouterr().out.count("Stopped ") == 5
    calls.clear()
    listed[:] = [("my-spindle", "stopped")]
    cli.main(["stop", "--app", "my-spindle"])
    assert calls == [["app", "list", "--json"]]
    assert len(deleted) == 2  # a leftover registry folder is deleted even so
    assert "No running apps for my-spindle" in capsys.readouterr().out


def test_add_reports_a_failed_deploy_and_how_to_retry(monkeypatch):
    handle = frontend_that_deploys(
        [{"apps": "deploying"}, {"apps": "failed", "apps_error": "no H200 quota"}]
    )
    mock_frontend_client(monkeypatch, handle)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    with pytest.raises(ValueError, match="no H200 quota.*same spindle add again"):
        cli.add_deployment("variant", base="qwen35-9b-lora-16k", overrides={})


def test_add_posts_again_when_no_deploy_is_running(monkeypatch):
    requests = []
    handle = frontend_that_deploys(
        [{"apps": "missing"}, {"apps": "deploying"}, {"apps": "ready"}], requests
    )
    mock_frontend_client(monkeypatch, handle)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    cli.add_deployment("variant", base="b", overrides={})
    assert [r.method for r in requests] == ["POST", "GET", "POST", "GET", "GET"]


def test_add_gives_up_after_its_timeout(monkeypatch):
    mock_frontend_client(
        monkeypatch, frontend_that_deploys([{"apps": "deploying"}] * 3)
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    with pytest.raises(ValueError, match="not ready after 0s"):
        cli.add_deployment("variant", base="b", overrides={}, timeout_s=0)


def test_add_rejects_a_frontend_that_does_not_deploy_apps(monkeypatch):
    mock_frontend_client(monkeypatch, frontend_that_deploys([{}]))
    with pytest.raises(ValueError, match="does not deploy apps"):
        cli.add_deployment("variant", base="b", overrides={})


def test_remove_lists_training_models_and_asks_before_forcing(monkeypatch, capsys):
    requests = []
    models = [{"model_id": "s:train:0", "session_id": "s", "created_at": 1.0}]

    def handle(request):
        requests.append(request)
        if request.url.params.get("force") == "true":
            return httpx.Response(200, json={"deployment": {"name": "variant"}})
        return httpx.Response(
            409, json={"error": "conflict", "message": "busy", "models": models}
        )

    mock_frontend_client(monkeypatch, handle)
    with pytest.raises(ValueError, match="--force"):
        cli.remove_deployment("variant", confirm=lambda question: False)
    assert "s:train:0 (session s)" in capsys.readouterr().out
    cli.remove_deployment("variant", confirm=lambda question: True)
    assert "Removed variant" in capsys.readouterr().out
    requests.clear()
    capsys.readouterr()
    cli.main(["remove", "variant", "--force"])
    assert [r.url.params.get("force") for r in requests] == [None, "true"]
    assert "Warning: removing variant ends training" in capsys.readouterr().out
    requests.clear()
    with pytest.raises(SystemExit):
        cli.main(["remove", "variant"])
    assert len(requests) == 1


def test_write_registry_tells_every_frontend_container_to_reread(monkeypatch):
    registry, shared = SimpleNamespace(), InMemoryKeyValueStore()

    async def no_files(prefix):
        return ()

    async def put_many(items):
        pass

    registry.list_keys, registry.put_many = no_files, put_many
    monkeypatch.setattr(deployment_configs, "frontend_app_id", lambda *a: "ap-1")
    monkeypatch.setattr(deployment_configs, "deployment_kv", lambda *a: registry)
    monkeypatch.setattr(deployment_configs, "shared_kv", lambda *a: shared)
    deployment_configs.write_registry(
        "spindle", "dev", [deployment().model_dump(mode="json")]
    )
    assert asyncio.run(shared.get(REGISTRY_VERSION_KEY))
