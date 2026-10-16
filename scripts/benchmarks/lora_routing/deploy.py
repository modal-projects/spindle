"""Deploy a benchmark arm, then allow its trainer to use any available GPU region."""

from copy import deepcopy
from pathlib import Path

import modal

from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig, load
from spindle.providers.modal.deployment_apps import build_rollout_app, build_trainer_app
from spindle.providers.modal.lora_pool import LoraPoolSpec


def main():
    deployment = DeploymentConfig.create(
        load(Path(__file__).with_name("deployment.py"))
    )
    deploy([deployment])
    trainer_platform = deepcopy(deployment.recipe.platform)
    trainer_platform["modal"]["region"] = None
    with modal.enable_output():
        build_trainer_app(deployment, trainer_platform)[0].deploy(
            environment_name="lilo-deploy"
        )
        # Start all inference replicas while the trainer initializes. This keeps
        # inference cold-start time outside the measured rollout comparison.
        build_rollout_app(deployment, LoraPoolSpec(deployment.definition_id))[0].deploy(
            environment_name="lilo-deploy"
        )


if __name__ == "__main__":
    main()
