"""Deploy the 9B SWE-Gym trainer and samplers.

``--init-run`` writes ``run/launch.json`` once. Later invocations, including
the supervisor's recovery deploys, refresh the apps and leave that file alone.
"""

import argparse
import json
import os
import time
from pathlib import Path

import modal

from recipe import Config
from spindle.deployments import DeploymentConfig
from spindle.providers.modal.lora_pool import LoraPoolSpec

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run"
SANDBOX_APP = "swe-gym-9b-256k-sandboxes"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STAGGER_SECONDS = 450
HOURS = 168
STEPS = 100


def environment_name(recipe):
    return recipe.platform["modal"]["environment"] or os.environ.get(
        "MODAL_ENVIRONMENT"
    )


def lookup_kwargs(environment):
    return {"environment_name": environment} if environment else {}


def owned_apps(recipe, definition, pool):
    return [
        recipe.platform["frontend"],
        definition.trainer_app_name,
        definition.inference_app_name,
        pool.app_name,
        SANDBOX_APP,
    ]


def deploy_apps(recipe, *, inference_only=False):
    from spindle.deployment_cli import deploy
    from spindle.providers.modal.deployment_apps import build_rollout_app

    definition = DeploymentConfig.create(recipe)
    pool = LoraPoolSpec(definition.definition_id)
    environment = environment_name(recipe)
    deploy(
        [definition],
        refresh_trainers=[] if inference_only else [recipe.name],
        refresh_inference=[recipe.name],
    )
    rollout, _server = build_rollout_app(definition, pool, recipe.platform)
    rollout.deploy(**lookup_kwargs(environment))
    lookup = lookup_kwargs(environment)
    modal.Function.from_name(
        definition.trainer_app_name, "trainer", **lookup
    ).update_autoscaler(max_containers=1)
    modal.Server.from_name(pool.app_name, "Server", **lookup).update_autoscaler(
        min_containers=recipe.inference_min_replicas,
        max_containers=recipe.inference_max_replicas,
    )
    modal.Function.from_name(
        recipe.platform["frontend"], "execute_sample", **lookup
    ).update_autoscaler(min_containers=4, max_containers=32)
    endpoint = modal.Function.from_name(recipe.platform["frontend"], "server", **lookup)
    return definition, pool, endpoint.get_web_url()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-run", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    recipe = Config()
    if args.init_run and (RUN / "launch.json").exists():
        raise SystemExit(
            "run/launch.json already exists; inspect it before launching again"
        )
    definition = DeploymentConfig.create(recipe)
    pool = LoraPoolSpec(definition.definition_id)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "resolved.json").write_text(definition.model_dump_json(indent=2) + "\n")
    with modal.enable_output():
        definition, pool, base_url = deploy_apps(
            recipe, inference_only=args.inference_only
        )
    if not args.init_run:
        print(json.dumps({"frontend": recipe.platform["frontend"], "init_run": False}))
        return
    now = time.time()
    manifest = {
        "model": recipe.model,
        "model_revision": MODEL_REVISION,
        "context_tokens": recipe.max_context_length,
        "rank": 32,
        "environment": environment_name(recipe),
        "steps_per_job": STEPS,
        "hours": HOURS,
        "clients": 8,
        "client_stagger_seconds": STAGGER_SECONDS,
        "base_url": base_url,
        "trainer_app": definition.trainer_app_name,
        "sampler_app": pool.app_name,
        "owned_apps": owned_apps(recipe, definition, pool),
        "monitored_apps": [definition.trainer_app_name, pool.app_name],
        "frontend": recipe.platform["frontend"],
        "definition_id": definition.definition_id,
        "sandbox_app": SANDBOX_APP,
        "validated_tasks": len(
            json.loads((ROOT / "data" / "validated_workload_ids.json").read_text())
        ),
        "absolute_deadline": now + HOURS * 3600,
        "launched_at": now,
    }
    (RUN / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"frontend": manifest["frontend"], "base_url": base_url, "init_run": True}
        )
    )


if __name__ == "__main__":
    main()
