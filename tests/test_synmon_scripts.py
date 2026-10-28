import importlib.util
import json
from pathlib import Path

from synthetic_monitoring.config_validation import (
    FRONTEND_NAME,
    SCRIPTS,
    script_argv,
    synmon_configs,
)

DAPO_LAUNCH = (
    Path(__file__).resolve().parents[1]
    / "examples/dapo-27b-4h200-12b200/launch_clients.py"
)


def _launch_clients():
    spec = importlib.util.spec_from_file_location("dapo_launch_clients", DAPO_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scripts_share_clients_and_steps():
    for name, spec in SCRIPTS.items():
        argv = script_argv(name, base_url="https://server.modal.run", output="/tmp/out")
        assert argv[0]
        assert argv[argv.index("--clients") + 1] == str(spec["clients"])
        assert argv[argv.index("--steps") + 1] == str(spec["steps"])


def test_dapo_and_swe_entrypoints():
    dapo = script_argv(
        "dapo-math-8", base_url="https://server.modal.run", output="/tmp/dapo"
    )
    assert dapo[1].endswith("/opt/dapo-math/launch_clients.py")
    assert dapo[dapo.index("--clients") + 1] == "8"
    assert dapo[dapo.index("--steps") + 1] == "5"
    assert dapo[dapo.index("--base-url") + 1] == "https://server.modal.run"
    assert dapo[dapo.index("--mode") + 1] == "gate"

    swe = script_argv("swe-gym", base_url="https://server.modal.run", output="/tmp/swe")
    assert swe[1].endswith("swe_gym/swe_gym.py")
    assert swe[swe.index("--clients") + 1] == "4"
    assert swe[swe.index("--steps") + 1] == "5"
    assert swe[swe.index("--output") + 1] == "/tmp/swe"
    assert swe[swe.index("--hours") + 1] == "8.0"


def test_dapo_steps_override(tmp_path):
    launch = _launch_clients()
    source = tmp_path / "config.json"
    source.write_text(json.dumps({"updates": 30, "model": "Qwen/Qwen3.8-27B"}))
    assert launch.steps_config(source, None) == source
    rewritten = json.loads(launch.steps_config(source, 5).read_text())
    assert rewritten["updates"] == 5
    assert rewritten["model"] == "Qwen/Qwen3.8-27B"
    assert json.loads(source.read_text())["updates"] == 30


def test_dapo_recipe_accepts_eight_clients():
    recipe = synmon_configs(["qwen38-27b-lora-16k"])[0]
    assert recipe.max_context_length == 32768
    assert recipe.trainer_max_clients_per_instance == 8
    assert recipe.miles_cfg["max_lora_slots"] == 8
    assert recipe.miles_cfg["max_tokens_per_gpu"] == 18432
    assert recipe.miles_cfg["tensor_model_parallel_size"] == 4
    assert recipe.platform["frontend"] == FRONTEND_NAME
    assert recipe.platform["modal"]["environment"] == "ajhinh-dev"

    swe = synmon_configs(["qwen35-9b-lora-128k"])[0]
    assert swe.name == "qwen35-9b-lora-128k"
    assert swe.max_context_length == 131_072
    assert swe.trainer_max_clients_per_instance == 8
