"""The SWE-Gym supervisor recipe keeps four B300s training and eight sampling."""

import runpy
from pathlib import Path

from spindle.deployments import DeploymentConfig

ROOT = Path(__file__).resolve().parents[2] / "examples" / "swe-gym-9b-256k"
CONFIG = runpy.run_path(str(ROOT / "recipe.py"))["config"]


def test_swe_gym_256k_layout():
    definition = DeploymentConfig.create(CONFIG)
    miles = definition.trainer_settings["miles"]
    serving = definition.inference_settings
    assert CONFIG.name == "swe-gym-9b-256k"
    assert CONFIG.model == "Qwen/Qwen3.5-9B"
    assert CONFIG.max_context_length == 262144
    assert CONFIG.trainer_gpu == "B300"
    assert CONFIG.trainer_gpus_per_node == 4
    assert CONFIG.trainer_max_instances == 1
    assert CONFIG.trainer_max_clients_per_instance == 8
    assert (
        CONFIG.inference_gpus_per_node,
        CONFIG.inference_min_replicas,
        CONFIG.inference_max_replicas,
    ) == (
        1,
        8,
        8,
    )
    assert CONFIG.inference_gpu == "B300"
    assert CONFIG.session_idle_timeout_s == 86400
    assert CONFIG.pool_idle_timeout_s == 86400
    assert CONFIG.platform["frontend"] == "spindle-swe-gym-9b-256k"
    assert CONFIG.platform["modal"]["environment"] is None
    assert miles["tensor_model_parallel_size"] == 1
    assert miles["context_parallel_size"] == 4
    assert miles["max_tokens_per_gpu"] == 65536
    assert miles["max_lora_slots"] == 8
    assert miles["align_sequences_to_parallel_layout"] is True
    assert serving["max_running_requests"] == 64
    assert serving["max_queued_requests"] == 128
    assert serving["lora_backend"] == "csgmv"
    assert serving["max_loaded_loras"] == 32
    assert serving["max_loras_per_batch"] == 16
    assert definition.trainer_app_name == "spindle-trainer-swe-gym-9b-256k"
    assert definition.inference_app_name == "spindle-inference-swe-gym-9b-256k"
