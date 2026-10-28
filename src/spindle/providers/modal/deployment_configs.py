"""Deployment configs and frontend-only platform settings."""

import json
import os
from pathlib import Path

import modal

from spindle.deployments import DeploymentConfig

from .kv import deployment_kv

CONFIGS_FILE_ENV = "SPINDLE_DEPLOYMENT_CONFIGS_FILE"
CONFIGS_IMAGE_PATH = "/root/spindle_deployment_configs.json"
PLATFORM_ENV = "SPINDLE_PLATFORM"
POOL_CONFIG_ENV = "SPINDLE_POOL_DEPLOYMENT"
DEPLOYMENT_ENV = "SPINDLE_DEPLOYMENT"
WORKER_ROLE_ENV = "SPINDLE_WORKER_ROLE"
FRONTEND_TIMEOUT_S = 60


def configs_from_env():
    path = os.environ.get(CONFIGS_FILE_ENV)
    if not path:
        raise ValueError(
            "Missing deployment configs. Use spindle deploy with your Python config files."
        )
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Deployment configs must be a nonempty list")
    return [DeploymentConfig.model_validate(row) for row in rows]


def platform_from_env():
    data = os.environ.get(PLATFORM_ENV)
    if not data:
        return configs_from_env()[0].recipe.platform
    return json.loads(data)


def _ask_frontend(frontend, name, environment, missing, *args):
    try:
        call = modal.Function.from_name(
            frontend, name, environment_name=environment
        ).spawn(*args)
    except modal.exception.NotFoundError:
        return missing
    try:
        return call.get(timeout=FRONTEND_TIMEOUT_S)
    except (TimeoutError, modal.exception.TimeoutError):
        call.cancel()
        raise ValueError(
            f"The deployed {frontend!r} frontend did not answer within "
            f"{FRONTEND_TIMEOUT_S}s. If its containers fail to start "
            f"(see `modal app logs {frontend}`), stop it with "
            f"`modal app stop {frontend}` and deploy again."
        ) from None


def deployed_configs(frontend, environment=None):
    """Read the configuration carried by the currently deployed frontend."""
    return _ask_frontend(frontend, "deployed_configs", environment, [])


def deployed_pools(frontend, environment=None):
    """Read pools owned by this frontend before updating its inference settings."""
    return _ask_frontend(frontend, "deployed_pools", environment, [])


def deployed_platform(frontend, environment=None):
    return _ask_frontend(frontend, "deployed_platform", environment, None)


def start_weights_download(frontend, environment, definition_id):
    """Start the base-weight download in a detached call; True while it runs."""
    return _ask_frontend(
        frontend, "start_weights_download", environment, None, definition_id
    )


def pool_config(definition_id):
    for deployment in configs_from_env():
        if deployment.definition_id == definition_id:
            return deployment
    added = deployment_kv().dictionary.get(definition_id)
    return None if added is None else DeploymentConfig.model_validate(added["config"])


def provision_pool(deployment, pool, platform):
    """Ask the inference app to create a pool using its original code."""
    provision = modal.Function.from_name(
        deployment.inference_app_name,
        "provision",
        environment_name=platform["modal"]["environment"],
    )
    return provision.remote(deployment.model_dump_json(), pool.as_dict())
