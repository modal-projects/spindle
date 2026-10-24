"""Per-run trainer app constructed in the provisioner's deployment subprocess."""

import os

from spindle.deployments import DeploymentConfig

from .deployment_apps import build_trainer_app
from .deployment_configs import TRAINER_CONFIG_ENV, platform_from_env

deployment = DeploymentConfig.model_validate_json(os.environ[TRAINER_CONFIG_ENV])
app, trainer = build_trainer_app(deployment, platform_from_env())
