"""Live RL on one unchanged eight-GPU trainer and eight inference replicas.

Only the frontend routing setting changes between arms. The explicit initialize
step deploys the measured inference image and permits trainer placement in any
region. Later policy changes must preserve both GPU deployment identities.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import modal

from cache_deployment import measured_image
from spindle.configs.qwen38_27b_lora_16k import Config as Parent
from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig
from spindle.inference.routing_simple import SIMPLE_POLICIES
from spindle.providers.modal.deployment_apps import build_rollout_app, build_trainer_app
from spindle.providers.modal.lora_pool import LoraPoolSpec
from supervise import write


class Config(Parent):
    max_context_length = 16384
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 16
    inference_gpu = "H200"
    inference_gpus_per_node = 1
    inference_cpu = 8
    inference_memory_mib = 65536
    inference_min_replicas = 8
    inference_max_replicas = 8
    inference_target_concurrency = 64
    inference_http2 = True
    inference_env = {"SGLANG_SPINDLE_ROUTING_OBSERVATIONS": "1"}
    session_idle_timeout_s = 7200
    pool_idle_timeout_s = 7200
    overrides = {
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.max_lora_slots": 16,
        "miles_cfg.max_tokens_per_gpu": 16384,
        "platform.modal.environment": "lilo-deploy",
        "platform.modal.region": "us-west",
        "platform.secrets.api": "spindle-swe-staggered-api",
        "platform.secrets.sampler_proxy": "lilo-proxy",
        "platform.storage.assets": "lilo-model-assets",
        "platform.storage.checkpoints": "routing27-checkpoints-1011",
        "platform.storage.bulletin": "routing27-bulletin-1011",
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 4096,
        "sglang_cfg.max_loras_per_batch": 8,
        "sglang_cfg.enable_lora_overlap_loading": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--policy",
        choices=["naive", "affinity", "work", "domains", "capacity", *SIMPLE_POLICIES],
        required=True,
    )
    parser.add_argument("--work-rates", type=Path, required=True)
    parser.add_argument("--trainer-gpu", choices=["H100", "H200"], default="H200")
    parser.add_argument("--trainer-tp", type=int, choices=[2, 4], default=2)
    parser.add_argument(
        "--region", choices=["us-west", "us-east", "any"], default="us-west"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write allocation/config manifests without deploying",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    initial = args.root / "initial-deployment.json"
    if initial.exists() == args.initialize:
        raise ValueError(
            "Initialize once, then switch policies on the existing deployment"
        )
    recipe = Config(
        name=f"routing27-live-{args.run}",
        inference_lora_routing=args.policy,
        trainer_gpu=args.trainer_gpu,
        inference_routing_work_rates=json.loads(args.work_rates.read_text()),
        overrides={
            "platform.frontend": f"spindle-routing27-live-{args.run}",
            "miles_cfg.tensor_model_parallel_size": args.trainer_tp,
            "platform.modal.region": None if args.region == "any" else args.region,
        },
    )
    deployment = DeploymentConfig.create(recipe)
    pool = LoraPoolSpec(deployment.definition_id)
    if not args.initialize:
        previous = DeploymentConfig.model_validate_json(initial.read_text())
        assert previous.trainer_identity() == deployment.trainer_identity()
        assert previous.inference_identity() == deployment.inference_identity()
    write(
        args.root / "owned-apps.json",
        [
            recipe.platform["frontend"],
            deployment.trainer_app_name,
            deployment.inference_app_name,
            pool.app_name,
        ],
    )
    write(
        args.root / f"deployment-{args.policy}.json", deployment.model_dump(mode="json")
    )
    if args.prepare_only:
        return
    monitor = json.loads((args.root / "monitor.json").read_text())
    if (
        time.time() - monitor["updated"] > 60
        or monitor["deadline"] - time.time() < 1800
    ):
        raise RuntimeError("Start the allocation watchdog before deploying live RL")
    deploy([deployment])
    if args.initialize:
        trainer_platform = deepcopy(recipe.platform)
        trainer_platform["modal"]["region"] = None
        with modal.enable_output():
            build_trainer_app(deployment, trainer_platform)[0].deploy(
                environment_name="lilo-deploy"
            )
            build_rollout_app(deployment, pool, image=measured_image())[0].deploy(
                environment_name="lilo-deploy"
            )
        write(initial, deployment.model_dump(mode="json"))
    endpoint = modal.Function.from_name(
        recipe.platform["frontend"], "server", environment_name="lilo-deploy"
    )
    write(
        args.root / "endpoint.json",
        {
            "base_url": endpoint.get_web_url(),
            "pool": pool.app_name,
            "policy": args.policy,
        },
    )


if __name__ == "__main__":
    main()
