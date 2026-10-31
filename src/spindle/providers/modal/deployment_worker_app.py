"""Trainer or inference app for an added deployment, built in a deploy subprocess."""

import json
import os

from spindle.deployments import DeploymentConfig
from spindle.providers.modal.fft_pool import FFTPoolSpec
from spindle.providers.modal.lora_pool import LoraPoolSpec

from .deployment_apps import (
    build_inference_app,
    build_rollout_app,
    build_trainer_app,
)
from .deployment_configs import (
    DEPLOYMENT_ENV,
    POOL_CONFIG_ENV,
    WORKER_ROLE_ENV,
    platform_from_env,
)

deployment = DeploymentConfig.model_validate_json(os.environ[DEPLOYMENT_ENV])
role = os.environ[WORKER_ROLE_ENV]
if role == "rollout":
    pool_data = json.loads(os.environ[POOL_CONFIG_ENV])
    pool = (
        LoraPoolSpec.from_dict(pool_data)
        if deployment.parameterization == "lora"
        else FFTPoolSpec.from_dict(pool_data)
    )
    app = build_rollout_app(deployment, pool, platform_from_env())[0]
else:
    build = {"trainer": build_trainer_app, "inference": build_inference_app}[role]
    app = build(deployment, platform_from_env())[0]
