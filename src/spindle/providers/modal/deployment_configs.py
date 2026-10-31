"""Deployment registry and frontend settings shared by Modal processes."""

from __future__ import annotations

import json
import os

import modal

from spindle.control_plane.deployments import FrontendSettings
from spindle.deployments import DeploymentConfig

from .kv import deployment_kv

FRONTEND_SETTINGS_ENV = "SPINDLE_FRONTEND_SETTINGS"
PLATFORM_ENV = "SPINDLE_PLATFORM"
POOL_CONFIG_ENV = "SPINDLE_POOL_DEPLOYMENT"
DEPLOYMENT_ENV = "SPINDLE_DEPLOYMENT"
WORKER_ROLE_ENV = "SPINDLE_WORKER_ROLE"
FRONTEND_TIMEOUT_S = 60


def frontend_settings_from_env() -> FrontendSettings:
    value = os.environ.get(FRONTEND_SETTINGS_ENV)
    if not value:
        raise ValueError(f"Missing {FRONTEND_SETTINGS_ENV}")
    return FrontendSettings(**json.loads(value))


def platform_from_env() -> dict:
    value = os.environ.get(FRONTEND_SETTINGS_ENV)
    if value:
        return frontend_settings_from_env().platform
    value = os.environ.get(PLATFORM_ENV)
    if value:
        return json.loads(value)
    raise ValueError(f"Missing {FRONTEND_SETTINGS_ENV}")


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
    return _ask_frontend(frontend, "deployed_configs", environment, [])


def seed_registry_rows(frontend, environment, rows):
    """Write ready rows straight into a deployed frontend's registry.

    Used before a frontend that still serves a baked-in configs file is
    redeployed, so its recipes keep serving through the swap and their
    already-deployed apps are not deployed again.
    """
    import asyncio
    import time

    app = modal.App.lookup(frontend, environment_name=environment)
    registry = deployment_kv(app.app_id)
    now = time.time()

    async def seed():
        for config in rows:
            name = config["recipe"]["name"]
            if await registry.get(name) is not None:
                continue
            await registry.put(
                name,
                {
                    "config": config,
                    "ready": config,
                    "state": "ready",
                    "attempt": "migrated",
                    "added_at": now,
                    "updated_at": now,
                    "ready_at": now,
                    "origin": "deploy",
                },
            )

    asyncio.run(seed())


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
    if row is None or row.get("ready") is None:
        return None
    return DeploymentConfig.model_validate(row["ready"])


def provision_pool(deployment, pool, platform):
    provision = modal.Function.from_name(
        deployment.inference_app_name,
        "provision",
        environment_name=platform["modal"]["environment"],
    )
    return provision.remote(deployment.model_dump_json(), pool.as_dict())
