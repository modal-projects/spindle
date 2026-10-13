"""Deploy eight H200s and assign all three policies to the same fixed pool."""

import argparse
import json
from pathlib import Path

import modal

from spindle.configs.qwen38_27b_lora_16k import Config as Parent
from spindle.deployments import DeploymentConfig
from spindle.providers.modal.deployment_apps import build_rollout_app
from spindle.providers.modal.lora_pool import LoraPoolSpec
from spindle.providers.modal.rollout_image import base_image


class Config(Parent):
    inference_min_replicas = 8
    inference_max_replicas = 8
    inference_memory_mib = 65536
    inference_cpu = 8
    inference_target_concurrency = 64
    inference_startup_timeout_s = 1200
    inference_env = {"SGLANG_SPINDLE_ROUTING_OBSERVATIONS": "1"}
    overrides = {
        "platform.modal.environment": "lilo-deploy",
        "platform.modal.region": "us-west",
        "platform.secrets.api": "spindle-swe-staggered-api",
        "platform.secrets.sampler_proxy": "lilo-proxy",
        "platform.storage.assets": "lilo-model-assets",
        "platform.storage.checkpoints": "routing27-checkpoints-1011",
        "platform.storage.bulletin": "routing27-bulletin-1011",
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 4096,
        "sglang_cfg.max_loras_per_batch": 8,
        "sglang_cfg.enable_lora_overlap_loading": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Copy Spindle explicitly: the production image mounts it at runtime, while
    # the benchmark-only endpoint must be part of this derived image.
    root = Path(__file__).resolve().parents[3]
    measured_image = (
        base_image.add_local_dir(
            root / "src/spindle", "/opt/cache-benchmark/spindle", copy=True
        )
        .add_local_file(
            Path(__file__).with_name("cache_instrumentation.py"),
            "/opt/cache-benchmark/instrument.py",
            copy=True,
        )
        .run_commands(
            "python /opt/cache-benchmark/instrument.py --sglang-root /sgl-workspace/sglang/python/sglang --sidecar /opt/cache-benchmark/spindle/inference/lora_sidecar.py"
        )
        .env({"PYTHONPATH": "/opt/cache-benchmark", "PYTHONSAFEPATH": "1"})
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with modal.enable_output():
        recipe = Config(name=f"cache27-{args.run}")
        deployment = DeploymentConfig.create(recipe)
        pool = LoraPoolSpec(deployment.definition_id)
        app, _ = build_rollout_app(deployment, pool, image=measured_image)
        app.deploy(environment_name="lilo-deploy")
    assignments = [
        {
            "arm": arm,
            "pool": pool.app_name,
            "deployment": json.loads(deployment.model_dump_json()),
        }
        for arm in ("default", "lora", "kv", "work", "work-no-adapter")
    ]
    args.output.write_text(json.dumps(assignments, indent=2) + "\n")


if __name__ == "__main__":
    main()
