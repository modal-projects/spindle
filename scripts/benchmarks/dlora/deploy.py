import argparse
import json
from pathlib import Path

import modal

from spindle.configs.qwen38_27b_lora_16k import Config as Parent
from spindle.deployments import DeploymentConfig
from spindle.inference.dlora_scheduler import MergeProfile
from spindle.providers.modal.deployment_apps import build_rollout_app
from spindle.providers.modal.lora_pool import LoraPoolSpec
from image import measured_image


class Config(Parent):
    inference_cpu = 16
    inference_memory_mib = 196608
    inference_target_concurrency = 36
    inference_http2 = True
    overrides = {
        "platform.modal.environment": "lilo-deploy",
        "platform.modal.region": "us-west",
        "platform.secrets.api": "spindle-swe-staggered-api",
        "platform.secrets.sampler_proxy": "lilo-proxy",
        "platform.storage.assets": "lilo-model-assets",
        "platform.storage.checkpoints": "routing27-checkpoints-1011",
        "platform.storage.bulletin": "routing27-bulletin-1011",
        "sglang_cfg.max_running_requests": 36,
        # The sidecar bounds total admitted work to 36 + 4. SGLang counts
        # arrivals before its scheduling pass, including free execution slots.
        "sglang_cfg.max_queued_requests": 40,
        "sglang_cfg.enable_lora_merge": True,
        "sglang_cfg.lora_unmerge_mode": "restore",
        "sglang_cfg.enable_lora_overlap_loading": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--region", choices=["us-west", "us-east"], default="us-west")
    parser.add_argument("--gpu", choices=["H100", "H200"], default="H200")
    parser.add_argument("--tensor-parallel", type=int, choices=[1, 2], default=1)
    parser.add_argument("--memory-mib", type=int, default=131072)
    parser.add_argument("--cpu", type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.replicas <= 8
    args.root.mkdir(parents=True, exist_ok=True)
    profile = vars(MergeProfile(0.03, 0.025, 8000, 2, 1, 36, enabled=False))
    config = Config(
        name=f"cache27-dlora-{args.run}",
        inference_min_replicas=args.replicas,
        inference_max_replicas=args.replicas,
        inference_gpu=args.gpu,
        inference_gpus_per_node=args.tensor_parallel,
        inference_cpu=args.cpu,
        inference_memory_mib=args.memory_mib,
        overrides={
            "platform.modal.region": args.region,
            "sglang_cfg.tp_size": args.tensor_parallel,
        },
        inference_env={
            "SGLANG_SPINDLE_ROUTING_OBSERVATIONS": "1",
            "SGLANG_SPINDLE_MERGE_PROFILE": json.dumps(profile),
        },
    )
    deployment = DeploymentConfig.create(config)
    pool = LoraPoolSpec(deployment.definition_id)
    (args.root / "pools.json").write_text(
        json.dumps(
            [
                dict(
                    arm=arm,
                    pool=pool.app_name,
                    deployment=json.loads(deployment.model_dump_json()),
                )
                for arm in ["kv", "dlora", "work", "dlora-no-merge"]
            ],
            indent=2,
        )
    )
    with modal.enable_output():
        app, _ = build_rollout_app(deployment, pool, image=measured_image())
        app.deploy(environment_name="lilo-deploy")


if __name__ == "__main__":
    main()
