"""Same current-default routing baseline and Rust candidate, with identical engines."""

import argparse
import json
from pathlib import Path

import modal
from scripts.benchmarks.rl_four_arm.deploy import Config as Parent
from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig
from spindle.providers.modal.lora_pool import LoraPoolSpec


class Config(Parent):
    trainer_gpu = "H200"
    inference_gpu = "H200"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=["default", "joint", "load_balanced", "lora_radix"],
        required=True,
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    directory = args.root / args.policy
    directory.mkdir(parents=True, exist_ok=True)
    # The root name is frozen in the manifest; never reuse another experiment's apps.
    name = f"routing27-rust-{args.policy}-{args.root.parent.name}-{args.root.name}"
    recipe = Config(
        name=name,
        inference_lora_routing="affinity" if args.policy == "default" else args.policy,
        inference_env=Config.inference_env | {"SGLANG_ENABLE_LORA_CSGMV_SPLIT_K": "1"},
        inference_routing_work_rates=json.loads(
            (args.root / "work-rates.json").read_text()
        ),
        overrides={"platform.frontend": f"spindle-{name}"},
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
