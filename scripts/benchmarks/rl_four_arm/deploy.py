"""Four isolated deployments comparing kernels and routing."""

import argparse
import json
from pathlib import Path

import modal

from spindle.configs.qwen38_27b_lora_16k import Config as Parent
from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig
from spindle.providers.modal.lora_pool import LoraPoolSpec


class Config(Parent):
    trainer_gpu = "B200"
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    inference_gpu = "B200"
    inference_gpus_per_node = 1
    inference_cpu = 8
    inference_memory_mib = 98304
    inference_min_replicas = 1
    inference_max_replicas = 4
    inference_target_concurrency = 16
    inference_scaledown_window_s = 300
    inference_env = {
        "SGLANG_SPINDLE_ROUTING_OBSERVATIONS": "1",
        "SGLANG_SPINDLE_MODEL_ROUTING": "1",
    }
    session_idle_timeout_s = 7200
    pool_idle_timeout_s = 7200
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.max_lora_slots": 8,
        "miles_cfg.max_tokens_per_gpu": 16384,
        "platform.modal.environment": "lilo-deploy",
        "platform.modal.region": None,
        "platform.secrets.api": "routing27-three-api-v2-1015",
        "platform.secrets.sampler_proxy": "lilo-proxy",
        "platform.storage.assets": "lilo-model-assets",
        "platform.storage.checkpoints": "routing27-checkpoints-1011",
        "platform.storage.bulletin": "routing27-bulletin-1011",
        "sglang_cfg.max_running_requests": 32,
        "sglang_cfg.max_queued_requests": 8,
        "sglang_cfg.max_loras_per_batch": 8,
        "sglang_cfg.enable_lora_overlap_loading": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=["default-no-split-k", "default", "stable", "model-spill"],
        required=True,
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    directory = args.root / args.policy
    directory.mkdir(parents=True, exist_ok=True)
    recipe = Config(
        name=f"routing27-four-{args.policy}-1015",
        inference_lora_routing="affinity"
        if args.policy in ("default", "default-no-split-k")
        else args.policy,
        inference_env=Config.inference_env
        | (
            {"SGLANG_ENABLE_LORA_CSGMV_SPLIT_K": "1"}
            if args.policy != "default-no-split-k"
            else {}
        ),
        inference_routing_work_rates=json.loads(
            (args.root / "work-rates.json").read_text()
        ),
        overrides={"platform.frontend": f"spindle-routing27-four-{args.policy}-1015"},
    )
    deployment = DeploymentConfig.create(recipe)
    pool = LoraPoolSpec(deployment.definition_id)
    owned = [
        recipe.platform["frontend"],
        deployment.trainer_app_name,
        deployment.inference_app_name,
        pool.app_name,
    ]
    (directory / "owned-apps.json").write_text(json.dumps(owned, indent=2))
    (directory / "deployment.json").write_text(deployment.model_dump_json(indent=2))
    if args.prepare_only:
        return
    deploy(
        [deployment],
        refresh_trainers=[deployment.name],
        refresh_inference=[deployment.name],
    )
    endpoint = modal.Function.from_name(
        recipe.platform["frontend"], "server", environment_name="lilo-deploy"
    )
    (directory / "endpoint.json").write_text(
        json.dumps(dict(base_url=endpoint.get_web_url(), pool=pool.app_name))
    )


if __name__ == "__main__":
    main()
