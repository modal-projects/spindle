"""The blog DAPO recipe keeps the 4×H200 trainer and 12×B200 sampler."""

import runpy
from pathlib import Path

from spindle.deployments import DeploymentConfig

ROOT = Path(__file__).resolve().parents[2] / "examples" / "dapo-27b-4h200-12b200"
DEPLOY = runpy.run_path(str(ROOT / "deploy.py"))
CONFIG = DEPLOY["config"]


def test_blog_gpu_split():
    definition = DeploymentConfig.create(CONFIG)
    miles = definition.trainer_settings["miles"]
    serving = definition.inference_settings
    assert CONFIG.trainer_gpu == "H200"
    assert CONFIG.trainer_gpus_per_node == 4
    assert CONFIG.trainer_nodes == 1
    assert CONFIG.inference_gpu == "B200"
    assert (CONFIG.inference_gpus_per_node, CONFIG.inference_min_replicas, CONFIG.inference_max_replicas) == (
        4,
        3,
        3,
    )
    assert miles["tensor_model_parallel_size"] == 1
    assert miles["max_tokens_per_gpu"] == 18432
    assert miles["max_lora_slots"] == 8
    assert serving["tp_size"] == 4
    assert serving["max_running_requests"] == 80
    assert serving["max_queued_requests"] == 8
    assert CONFIG.inference_target_concurrency == 80
    assert CONFIG.inference_lora_routing == "cache_wait"
    assert CONFIG.inference_routing_balance_absolute == 32
    assert CONFIG.inference_routing_balance_relative == 2.0
    assert CONFIG.inference_routing_cache_threshold == 0.5


def test_sweep_slot_override_matches_client_capacity():
    recipe = type(CONFIG)()
    DEPLOY["apply_lora_slots"](recipe, 16)
    definition = DeploymentConfig.create(recipe)
    assert definition.trainer_settings["miles"]["max_lora_slots"] == 16
    assert definition.inference_settings["max_loras_per_batch"] == 16
    assert recipe.trainer_max_clients_per_instance == 16
