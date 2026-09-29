"""Generic rollout app constructed in the pool deployment subprocess."""

import os

from spindle.deployments import DeploymentConfig

from .deployment_apps import build_rollout_app
from .deployment_configs import POOL_CONFIG_ENV, platform_from_env
from .fft_pool import FFTPoolSpec
from .lora_pool import LoraPoolSpec

deployment = DeploymentConfig.model_validate_json(os.environ[POOL_CONFIG_ENV])
if deployment.parameterization == "lora":
    pool = LoraPoolSpec(deployment.definition_id)
else:
    pool = FFTPoolSpec(
        definition_id=deployment.definition_id,
        model_id=os.environ["SPINDLE_FFT_POOL_MODEL_ID"],
        latest=os.environ["SPINDLE_FFT_POOL_LATEST"] == "1",
        version=int(os.environ["SPINDLE_FFT_POOL_VERSION"]),
        **{
            key: int(os.environ[f"SPINDLE_FFT_POOL_{key.upper()}"])
            for key in ("min_containers", "max_containers", "scaledown_window")
            if f"SPINDLE_FFT_POOL_{key.upper()}" in os.environ
        },
    )
app, Server = build_rollout_app(deployment, pool, platform_from_env())
