"""4×H200 trainer and 12×B200 sampler for the Qwen3.8-27B DAPO run.

This is the compute layout from the Training API post: one trainer node and
three TP4 sampling replicas. The client loop is client.py.
"""

import argparse
import json
from pathlib import Path

import modal

from spindle.configs.qwen38_27b_lora_16k import Config as Parent
from spindle.deployments import DeploymentConfig
from spindle.providers.modal.lora_pool import LoraPoolSpec

ROOT = Path(__file__).resolve().parent


class Config(Parent):
    name = "dapo27-4h200-12b200"
    max_context_length = 32768
    trainer_gpu = "H200"
    trainer_gpus_per_node = 4
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    trainer_timeout_s = 43200
    inference_gpu = "B200"
    inference_gpus_per_node = 4
    inference_cpu = 16
    inference_memory_mib = 196608
    inference_min_replicas = 3
    inference_max_replicas = 3
    # Three TP4 replicas share twelve inference GPUs; the trainer uses four.
    inference_target_concurrency = 80
    # Blog routing policy. The GPU split and sglang_cfg limits are what deploy
    # applies; cache_wait is the policy used for that run.
    inference_lora_routing = "cache_wait"
    inference_routing_balance_absolute = 32
    inference_routing_balance_relative = 2.0
    inference_routing_cache_threshold = 0.5
    inference_routing_prefill_tokens_per_second = 10463.96139191663
    session_idle_timeout_s = 43200
    pool_idle_timeout_s = 43200
    overrides = {
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 8,
        # A 32,768-token packing trial OOMed. Keep the full optimizer batch
        # and pack it into smaller microbatches.
        "miles_cfg.max_tokens_per_gpu": 18432,
        "platform.frontend": "spindle-dapo27-4h200-12b200",
        "platform.storage.checkpoints": "dapo27-4h200-12b200-checkpoints",
        "platform.storage.bulletin": "dapo27-4h200-12b200-bulletin",
        "sglang_cfg.tp_size": 4,
        "sglang_cfg.max_running_requests": 80,
        "sglang_cfg.max_queued_requests": 8,
        "sglang_cfg.max_loras_per_batch": 8,
        "sglang_cfg.max_loaded_loras": 256,
        "sglang_cfg.enable_lora_overlap_loading": False,
        "sglang_cfg.mem_fraction_static": 0.85,
        "sglang_cfg.chunked_prefill_size": 8192,
        "sglang_cfg.cuda_graph_max_bs_decode": 96,
        "sglang_cfg.enable_metrics": True,
    }


config = Config()


def apply_lora_slots(recipe, slots):
    if slots < 1:
        raise ValueError("lora slots must be positive")
    recipe.miles_cfg["max_lora_slots"] = slots
    recipe.sglang_cfg["max_loras_per_batch"] = slots
    recipe.trainer_max_clients_per_instance = slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--lora-slots",
        type=int,
        default=8,
        help="8 matches the two-client blog run; the 1/2/4/8 client sweep used 16",
    )
    args = parser.parse_args()
    apply_lora_slots(config, args.lora_slots)
    definition = DeploymentConfig.create(config)
    pool = LoraPoolSpec(definition.definition_id)
    owned = [
        config.platform["frontend"],
        definition.trainer_app_name,
        definition.inference_app_name,
        pool.app_name,
    ]
    (ROOT / "deployment.json").write_text(definition.model_dump_json(indent=2) + "\n")
    (ROOT / "owned-apps.json").write_text(json.dumps(owned, indent=2) + "\n")
    if args.launch:
        from spindle.deployment_cli import deploy

        deploy(
            [definition],
            refresh_trainers=[definition.name],
            refresh_inference=[definition.name],
        )
        environment = config.platform["modal"]["environment"]
        endpoint = modal.Function.from_name(
            owned[0],
            "server",
            **({"environment_name": environment} if environment else {}),
        )
        (ROOT / "endpoint.json").write_text(
            json.dumps(
                {"base_url": endpoint.get_web_url(), "pool": pool.app_name}, indent=2
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "apps": owned,
                "trainer_gpus": 4,
                "inference_gpus": 12,
                "launched": args.launch,
            }
        )
    )


if __name__ == "__main__":
    main()
