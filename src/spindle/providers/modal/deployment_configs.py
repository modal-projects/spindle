"""Deployment configs and frontend-only platform settings."""

import json
import os

import modal

from spindle.deployments import DeploymentConfig

CONFIGS_ENV = "SPINDLE_DEPLOYMENT_CONFIGS"
PLATFORM_ENV = "SPINDLE_PLATFORM"
POOL_CONFIG_ENV = "SPINDLE_POOL_DEPLOYMENT"


def configs_from_env():
    data = os.environ.get(CONFIGS_ENV)
    if not data:
        raise ValueError(
            "Missing deployment configs. Use spindle deploy with your Python config files."
        )
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Deployment configs must be a nonempty list")
    return [DeploymentConfig.model_validate(row) for row in rows]


def platform_from_env():
    data = os.environ.get(PLATFORM_ENV)
    if not data:
        return configs_from_env()[0].recipe.platform
    return json.loads(data)


def deployed_configs(frontend, environment=None):
    """Read the configuration carried by the currently deployed frontend."""
    try:
        return modal.Function.from_name(
            frontend, "deployed_configs", environment_name=environment
        ).remote()
    except modal.exception.NotFoundError:
        return []


def deployed_pools(frontend, environment=None):
    """Read pools owned by this frontend before updating its inference settings."""
    try:
        return modal.Function.from_name(
            frontend, "deployed_pools", environment_name=environment
        ).remote()
    except modal.exception.NotFoundError:
        return []


def deployed_platform(frontend, environment=None):
    try:
        return modal.Function.from_name(
            frontend, "deployed_platform", environment_name=environment
        ).remote()
    except modal.exception.NotFoundError:
        return None


def pool_config(definition_id):
    for deployment in configs_from_env():
        if deployment.definition_id == definition_id:
            return deployment
    return None


def provision_pool(deployment, pool, platform):
    """Ask the inference app to create a pool using its original code."""
    provision = modal.Function.from_name(
        deployment.inference_app_name,
        "provision",
        environment_name=platform["modal"]["environment"],
    )
    return provision.remote(deployment.model_dump_json(), pool.as_dict())
