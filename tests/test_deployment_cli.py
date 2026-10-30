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
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal import deployment_configs
from spindle.providers.modal.deployment_configs import (
    FRONTEND_SETTINGS_ENV,
    deployed_configs,
)
from spindle.providers.modal.image_dependencies import ignore_config_source


def deployment(preset="qwen35-9b-lora-16k"):
    return DeploymentConfig.create(load(config_path(preset)))


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setattr(cli.sys, "version_info", (3, 12, 0))
    state = SimpleNamespace(
        legacy=[],
        seeded=None,
        rows={
            "unlisted": {
                "config": {"recipe": {"name": "unlisted"}},
                "state": "ready",
            }
        },
        platform=None,
        calls=[],
        fail=None,
        settings=None,
        env=None,
        configs=None,
        refresh=None,
        images=[],
    )

    def run(command, *, env, check, **kwargs):
        assert command[command.index("--tag") + 1]
        state.calls.append("frontend")
        if state.fail == "frontend":
            raise subprocess.CalledProcessError(1, command)
        state.settings = json.loads(env[FRONTEND_SETTINGS_ENV])
        state.env = env
        state.platform = state.settings["platform"]
        state.configs = None
        return SimpleNamespace(stdout="", stderr="")

    class Function:
        def remote(self, configs_json, refresh_trainers, refresh_inference):
            state.calls.append("deploy_recipes")
            state.legacy = []
            state.configs = json.loads(configs_json)
            state.refresh = (refresh_trainers, refresh_inference)
            for config in state.configs:
                state.rows[config["recipe"]["name"]] = {
                    "config": config,
                    "ready": config,
                    "state": "failed"
                    if state.fail == config["recipe"]["name"]
                    else "ready",
                    "error": (
                        "trainer app deploy failed: no capacity"
                        if state.fail == config["recipe"]["name"]
                        else None
                    ),
                }

    monkeypatch.setattr(cli, "deployed_platform", lambda *args: state.platform)
    monkeypatch.setattr(
        cli,
        "deployed_configs",
        lambda *args: state.legacy or list(state.rows.values()),
    )

    def seed(frontend, environment, rows):
        state.calls.append("seed")
        state.seeded = rows
        state.legacy = []
        for config in rows:
            state.rows[config["recipe"]["name"]] = {
                "config": config,
                "ready": config,
                "state": "ready",
            }

    monkeypatch.setattr(cli, "seed_registry_rows", seed)
    monkeypatch.setattr(modal.Function, "from_name", lambda *args, **kwargs: Function())
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        cli,
        "build_sampler_image",
        lambda frontend, environment: state.images.append((frontend, environment)),
    )
    monkeypatch.setattr(cli.modal, "enable_output", nullcontext)
    return state


def test_deploy_migrates_every_recipe_of_a_pre_registry_frontend(deployed):
    unchanged = deployment()
    changed = deployment("qwen35-9b-fft-64k")
    served_before = changed.model_dump(mode="json")
    served_before["recipe"]["trainer_gpu"] = "B200"
    unlisted = deployment("qwen35-9b-lora-16k").model_dump(mode="json")
    unlisted["recipe"]["name"] = "unlisted-but-served"
    deployed.legacy = [unchanged.model_dump(mode="json"), served_before, unlisted]
    cli.deploy([unchanged, changed])
    assert deployed.calls == ["seed", "frontend", "deploy_recipes"]
    assert [row["recipe"]["name"] for row in deployed.seeded] == [
        unchanged.name,
        "unlisted-but-served",
    ]


def test_deploy_does_not_seed_a_recipe_being_refreshed(deployed):
    recipe = deployment()
    deployed.legacy = [recipe.model_dump(mode="json")]
    cli.deploy([recipe], refresh_trainers=[recipe.name])
    assert "seed" not in deployed.calls


def test_deploy_does_not_seed_a_registry_frontend(deployed):
    cli.deploy([deployment()])
    assert "seed" not in deployed.calls


def test_deploy_uses_frontend_settings_and_upserts_registry_rows(deployed):
    cli.deploy([deployment()])
    assert deployed.calls == ["frontend", "deploy_recipes"]
    assert FRONTEND_SETTINGS_ENV in deployed.env
    assert deployed.settings["platform"] == deployment().recipe.platform
    assert "unlisted" in deployed.rows
    assert deployed.rows["qwen35-9b-lora-16k"]["state"] == "ready"


def mock_add_client(monkeypatch, handler):
    client_type = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kwargs: client_type(transport=transport, **kwargs),
    )


def test_add_posts_base_and_json_parsed_overrides(monkeypatch, capsys):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(202, json={"deployment": {"state": "ready"}})

    mock_add_client(monkeypatch, handle)
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example/")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    cli.main(
        [
            "deploy",
            "--add",
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
    assert "Added variant." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--not_a_setting=1"], "unknown recipe field"),
        (["--trainer_gpu=H100", "--trainer_gpu=B200"], "duplicate field"),
        (["--trainer_gpu"], "missing value"),
        (["--=1"], "field names cannot be empty"),
        (["extra"], "cannot be combined with config files"),
        (["config.py"], "cannot be combined with config files"),
        (["--all"], "cannot be combined with config files"),
        (["--app", "frontend"], "cannot be combined with config files"),
        (["--env", "dev"], "cannot be combined with config files"),
        (["--region", "us-east"], "cannot be combined with config files"),
        (["--refresh-trainer", "base"], "cannot be combined with config files"),
        (["--refresh-inference", "base"], "cannot be combined with config files"),
        (["--set", "gpu=H200"], "unknown recipe field"),
    ],
)
def test_add_cli_validation_fails_before_http(extra_args, message, monkeypatch, capsys):
    requests = []
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: requests.append(kwargs))
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy", "--add", "variant", "--base", "base", *extra_args])
    assert exc.value.code == 1
    assert message in capsys.readouterr().err
    assert requests == []


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["deploy", "--add", "variant"], 1, "--base is required"),
        (["deploy", "--add", "", "--base", "base"], 1, "non-empty name"),
        (["deploy", "--base", "base"], 1, "--base requires --add"),
        (["deploy", "--add", "variant", "--base"], 2, "expected one argument"),
    ],
)
def test_add_cli_requires_name_and_base(args, code, message, monkeypatch, capsys):
    requests = []
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: requests.append(kwargs))
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == code
    assert message in capsys.readouterr().err
    assert requests == []


def test_field_flag_without_add_is_an_argparse_error(monkeypatch):
    requests = []
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: requests.append(kwargs))
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy", "--trainer_gpu=H200"])
    assert exc.value.code == 2
    assert requests == []


@pytest.mark.parametrize("missing", ["TINKER_BASE_URL", "TINKER_API_KEY"])
def test_add_cli_requires_frontend_credentials_before_http(missing, monkeypatch):
    requests = []
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: requests.append(kwargs))
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy", "--add", "variant", "--base", "base"])
    assert exc.value.code == 1
    assert requests == []


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (
            400,
            {"error": "invalid_request", "message": "unknown base deployment"},
            "unknown base deployment",
        ),
        (405, {"message": "method not allowed"}, "frontend predates adding configs"),
    ],
)
def test_add_http_errors_exit_without_polling(
    status, body, message, monkeypatch, capsys
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json=body)

    mock_add_client(monkeypatch, handle)
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy", "--add", "variant", "--base", "base"])
    assert exc.value.code == 1
    assert len(requests) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("final_state", "expected_error"),
    [
        ("ready", None),
        ("failed", "inference app deploy failed: no capacity"),
        ("stale", "frontend platform changed"),
    ],
)
def test_add_polls_until_ready_or_failure(
    final_state, expected_error, monkeypatch, capsys
):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"deployment": {"state": "deploying"}})
        return httpx.Response(
            200,
            json={
                "deployments": [
                    {
                        "name": "variant",
                        "state": final_state,
                        "error": expected_error,
                    }
                ]
            },
        )

    mock_add_client(monkeypatch, handle)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    args = ["deploy", "--add", "variant", "--base", "base"]
    if expected_error:
        with pytest.raises(SystemExit):
            cli.main(args)
        assert expected_error in capsys.readouterr().err
    else:
        cli.main(args)
        assert "Added variant." in capsys.readouterr().out
    assert [request.method for request in requests] == ["POST", "GET"]


def test_add_polling_times_out_without_a_second_request(monkeypatch, capsys):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(202, json={"deployment": {"state": "deploying"}})

    mock_add_client(monkeypatch, handle)
    monkeypatch.setattr(cli.time, "monotonic", Mock(side_effect=[0, 4201]))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setenv("TINKER_BASE_URL", "https://frontend.example")
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy", "--add", "variant", "--base", "base"])
    assert exc.value.code == 1
    assert "still deploying" in capsys.readouterr().err
    assert [request.method for request in requests] == ["POST"]


def test_deploy_passes_refresh_sets_and_platform_change_forces_both(deployed):
    row = deployment()
    cli.deploy([row], refresh_trainers=[row.name])
    assert deployed.refresh == ([row.name], [])

    previous = deepcopy(row.recipe.platform)
    deployed.platform = previous
    row.recipe.platform["modal"]["region"] = "us-east"
    cli.deploy([row])
    assert deployed.refresh == ([row.name], [row.name])


def test_deploy_raises_the_registry_error_on_failure(deployed):
    row = deployment()
    deployed.fail = row.name
    with pytest.raises(ValueError, match="trainer app deploy failed: no capacity"):
        cli.deploy([row])


def test_read_configs_from_deployed_function(monkeypatch):
    call = SimpleNamespace(get=lambda timeout: [{"config": {"recipe": {"name": "x"}}}])
    function = SimpleNamespace(spawn=lambda: call)
    monkeypatch.setattr(modal.Function, "from_name", lambda *a, **k: function)
    assert deployed_configs("my-app", "dev") == [{"config": {"recipe": {"name": "x"}}}]


def test_deploy_all_deploys_bundled_configs(tmp_path, monkeypatch, deployed):
    bundled = config_path("qwen35-9b-lora-16k").parent.glob("[!_]*.py")
    expected = {load(path).name for path in bundled}
    monkeypatch.chdir(tmp_path)
    cli.main(["deploy", "--all"])
    assert len(expected) > 1
    assert {row["recipe"]["name"] for row in deployed.configs} == expected
    assert max(len(value) for value in deployed.env.values()) < 131_072


def test_unresponsive_frontend_fails_fast(monkeypatch):
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
    assert deployed.calls == ["frontend", "deploy_recipes"]
    assert shown == [picked]


def test_progress_run_names_every_recipe_and_the_retry_command(monkeypatch, deployed):
    console = Console(record=True, width=120)
    presets = deploy_wizard.bundled_presets()
    picked = [deploy_wizard.pick(presets, "Qwen/Qwen3.5-9B-Base", "lora", 16_384)]
    deployed.fail = "qwen35-9b-lora-16k"
    with pytest.raises(ValueError, match="trainer app deploy failed"):
        cli.deploy(
            cli.compile_configs([preset.path for preset in picked]),
            run_steps=partial(
                deploy_wizard.deploy_with_progress,
                console,
                retry_command=deploy_wizard.deploy_command(picked),
            ),
        )
    output = console.export_text()
    assert "API server" in output
    assert "Recipe     qwen35-9b-lora-16k failed" in output
    assert "spindle deploy qwen35-9b-lora-16k" in output
    assert deployed.calls == ["frontend", "deploy_recipes"]


def test_sampler_image_failure_still_deploys_the_frontend(
    monkeypatch, deployed, capsys
):
    def broken(frontend, environment):
        raise RuntimeError("image build failed")

    monkeypatch.setattr(cli, "build_sampler_image", broken)
    cli.deploy([deployment()])
    assert deployed.calls == ["frontend", "deploy_recipes"]
    assert "Sampler image build failed: image build failed" in capsys.readouterr().err


def test_frontend_deploys_while_the_sampler_image_builds(monkeypatch, deployed):
    frontend_deployed = threading.Event()
    deploy_frontend = subprocess.run

    def build(frontend, environment):
        assert frontend_deployed.wait(timeout=5)
        deployed.images.append(frontend)

    def run(*args, **kwargs):
        result = deploy_frontend(*args, **kwargs)
        frontend_deployed.set()
        return result

    monkeypatch.setattr(cli, "build_sampler_image", build)
    monkeypatch.setattr(subprocess, "run", run)
    cli.deploy([deployment()])
    assert deployed.calls == ["frontend", "deploy_recipes"]
    assert deployed.images == ["spindle"]


def test_cancelled_wizard_deploys_nothing(monkeypatch, deployed, capsys):
    def cancel(*args):
        raise deploy_wizard.Cancelled

    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr(deploy_wizard, "choose", cancel)
    with pytest.raises(SystemExit):
        cli.main(["deploy"])
    assert "cancelled" in capsys.readouterr().err
    assert deployed.calls == []


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
        modal.Function,
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
