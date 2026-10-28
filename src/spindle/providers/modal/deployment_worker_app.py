"""Trainer or inference app for an added deployment, built in a deploy subprocess."""

import os

from spindle.deployments import DeploymentConfig

from .deployment_apps import build_inference_app, build_trainer_app
from .deployment_configs import DEPLOYMENT_ENV, WORKER_ROLE_ENV, platform_from_env

deployment = DeploymentConfig.model_validate_json(os.environ[DEPLOYMENT_ENV])
build = {"trainer": build_trainer_app, "inference": build_inference_app}[
    os.environ[WORKER_ROLE_ENV]
]
app = build(deployment, platform_from_env())[0]
