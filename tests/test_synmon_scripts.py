import importlib.util
import json
from pathlib import Path

from synthetic_monitoring.config_validation import (
    FRONTEND_NAME,
    SCRIPTS,
    light_dapo_config,
    recipes_for,
    script_argv,
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
        "dapo-math",
        base_url="https://server.modal.run",
        output="/tmp/dapo",
        config="/tmp/dapo/config.json",
    )
    assert dapo[1].endswith("/opt/dapo-math/launch_clients.py")
    assert dapo[dapo.index("--clients") + 1] == "2"
    assert dapo[dapo.index("--steps") + 1] == "5"
    assert dapo[dapo.index("--base-url") + 1] == "https://server.modal.run"
    assert dapo[dapo.index("--config") + 1] == "/tmp/dapo/config.json"
    assert dapo[dapo.index("--mode") + 1] == "gate"

    swe = script_argv("swe-gym", base_url="https://server.modal.run", output="/tmp/swe")
    assert swe[1].endswith("swe_gym/swe_gym.py")
    assert swe[swe.index("--clients") + 1] == "2"
    assert swe[swe.index("--steps") + 1] == "5"
    assert swe[swe.index("--groups-per-batch") + 1] == "1"
    assert swe[swe.index("--output") + 1] == "/tmp/swe"
    assert swe[swe.index("--hours") + 1] == "6.0"


def test_dapo_steps_override(tmp_path):
    launch = _launch_clients()
    source = tmp_path / "config.json"
    source.write_text(json.dumps({"updates": 30, "model": "Qwen/Qwen3.8-27B"}))
    assert launch.steps_config(source, None) == source
    rewritten = json.loads(launch.steps_config(source, 5).read_text())
    assert rewritten["updates"] == 5
    assert rewritten["model"] == "Qwen/Qwen3.8-27B"
    assert json.loads(source.read_text())["updates"] == 30


def test_light_dapo_config_shrinks_the_step():
    cfg = light_dapo_config(
        {
            "updates": 30,
            "context_tokens": 32768,
            "max_tokens": 16384,
            "groups_per_update": 8,
        }
    )
    assert cfg["updates"] == 30
    assert cfg["context_tokens"] == 8192
    assert cfg["max_tokens"] == 1024
    assert cfg["max_tokens"] < cfg["context_tokens"]
    assert cfg["groups_per_update"] == 1
    assert cfg["samples_per_group"] == 4


def _assert_inference_cap(recipe):
    assert recipe.inference_gpu == "B200"
    assert recipe.inference_gpus_per_node * recipe.inference_max_replicas == 2
    assert recipe.inference_min_replicas <= recipe.inference_max_replicas
    assert recipe.sglang_cfg["tp_size"] == 1
    assert recipe.platform["frontend"] == FRONTEND_NAME


def test_dapo_trainer_stays_on_two_h200():
    recipe = recipes_for("dapo-math")[0]
    assert recipe.name == "qwen38-27b-lora-16k"
    assert recipe.trainer_gpu == "H200"
    assert recipe.trainer_gpus_per_node == 2
    assert recipe.trainer_nodes == 1
    assert recipe.max_context_length == 8192
    assert recipe.miles_cfg["tensor_model_parallel_size"] == 2
    assert recipe.miles_cfg["max_tokens_per_gpu"] == 8192
    _assert_inference_cap(recipe)


def test_swe_trainer_fits_128k_in_32k_microbatches():
    recipe = recipes_for("swe-gym")[0]
    assert recipe.name == "qwen35-9b-lora-128k"
    assert recipe.trainer_gpu == "H200"
    assert recipe.trainer_gpus_per_node == 4
    assert recipe.trainer_nodes == 1
    assert recipe.max_context_length == 131_072
    miles = recipe.miles_cfg
    assert miles["tensor_model_parallel_size"] == 1
    assert miles["context_parallel_size"] == 4
    assert miles["max_tokens_per_gpu"] == 32768
    assert (
        miles["context_parallel_size"] * miles["max_tokens_per_gpu"]
        >= recipe.max_context_length
    )
    assert miles["align_sequences_to_parallel_layout"] is True
    assert recipe.sglang_cfg["chunked_prefill_size"] == 8192
    _assert_inference_cap(recipe)
