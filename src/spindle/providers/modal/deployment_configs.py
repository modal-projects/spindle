"""Deployment registry and frontend settings shared by Modal processes."""

from __future__ import annotations

import asyncio
import json
import os

import modal

from spindle.control_plane.deployments import FrontendSettings, mark_registry_changed
from spindle.deployments import DeploymentConfig

from .kv import deployment_kv, shared_kv

FRONTEND_SETTINGS_ENV = "SPINDLE_FRONTEND_SETTINGS"
PLATFORM_ENV = "SPINDLE_PLATFORM"
POOL_CONFIG_ENV = "SPINDLE_POOL_DEPLOYMENT"
WORKER_CONFIG_ENV = "SPINDLE_WORKER_DEPLOYMENT"
WORKER_ROLE_ENV = "SPINDLE_WORKER_ROLE"
FRONTEND_TIMEOUT_S = 60


def frontend_settings_from_env() -> FrontendSettings:
    value = os.environ.get(FRONTEND_SETTINGS_ENV)
    if not value:
        raise ValueError(f"Missing {FRONTEND_SETTINGS_ENV}")
    return FrontendSettings(**json.loads(value))


def platform_from_env() -> dict:
    if os.environ.get(FRONTEND_SETTINGS_ENV):
        return frontend_settings_from_env().platform
    value = os.environ.get(PLATFORM_ENV)
    if not value:
        raise ValueError(f"Missing {FRONTEND_SETTINGS_ENV}")
    return json.loads(value)


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
    app_id = frontend_app_id(frontend, environment)
    if app_id is None:
        return []
    items = asyncio.run(deployment_kv(app_id, environment).list_items(""))
    if not items:
        return _ask_frontend(frontend, "deployed_configs", environment, [])
    return [row for _, row in items]


def write_registry(frontend, environment, configs, *, keep_existing=()):
    app_id = frontend_app_id(frontend, environment)
    if app_id is None:
        return False
    registry = deployment_kv(app_id, environment)

    async def write():
        existing = set(await registry.list_keys(""))
        items = {
            config["recipe"]["name"]: config
            for config in keep_existing
            if config["recipe"]["name"] not in existing
        }
        items.update((config["recipe"]["name"], config) for config in configs)
        await registry.put_many(items)
        await mark_registry_changed(shared_kv(app_id, environment))

    asyncio.run(write())
    return True


def frontend_app_id(frontend, environment=None):
    try:
        return modal.App.lookup(frontend, environment_name=environment).app_id
    except modal.exception.NotFoundError:
        return None


def delete_registry(app_id, environment=None):
    asyncio.run(deployment_kv(app_id, environment).clear())


def deployed_pools(frontend, environment=None):
    return _ask_frontend(frontend, "deployed_pools", environment, [])


def deployed_platform(frontend, environment=None):
    return _ask_frontend(frontend, "deployed_platform", environment, None)


def start_weights_download(frontend, environment, definition_id):
    return _ask_frontend(
        frontend, "start_weights_download", environment, None, definition_id
    )


def pool_config(definition_id):
    row = deployment_kv().read(definition_id)
    return None if row is None else DeploymentConfig.from_registry(row)


def provision_pool(deployment, pool, platform):
    provision = modal.Function.from_name(
        deployment.inference_app_name,
        "provision",
        environment_name=platform["modal"]["environment"],
    )
    return provision.remote(deployment.model_dump_json(), pool.as_dict())
