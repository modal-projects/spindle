"""Deploy a benchmark arm, then allow its trainer to use any US GPU region."""

from copy import deepcopy
from pathlib import Path

import modal

from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig, load
from spindle.providers.modal.deployment_apps import build_trainer_app


def main():
    deployment = DeploymentConfig.create(
        load(Path(__file__).with_name("deployment.py"))
    )
    deploy([deployment])
    trainer_platform = deepcopy(deployment.recipe.platform)
    trainer_platform["modal"]["region"] = "us"
    with modal.enable_output():
        build_trainer_app(deployment, trainer_platform)[0].deploy(
            environment_name="lilo-deploy"
        )


if __name__ == "__main__":
    main()
