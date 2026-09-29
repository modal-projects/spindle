from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import modal
from stitch.pools.modal_flash import ModalFlashPool

from .deployment_configs import (
    PLATFORM_ENV,
    POOL_CONFIG_ENV,
    platform_from_env,
    pool_config,
    provision_pool,
)


@dataclass(frozen=True)
class LoraPoolSpec:
    definition_id: str

    def __post_init__(self) -> None:
        if Path(self.definition_id).name != self.definition_id:
            raise ValueError(f"invalid definition id: {self.definition_id!r}")

    @classmethod
    def from_dict(cls, value: dict) -> LoraPoolSpec:
        return cls(definition_id=str(value["definition_id"]))

    @property
    def app_name(self) -> str:
        return f"spindle-lora-{self.definition_id}"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def env(self) -> dict[str, str]:
        return {
            "SPINDLE_LORA_POOL_APP_NAME": self.app_name,
            "SPINDLE_LORA_POOL_DEFINITION_ID": self.definition_id,
        }


async def pool_gateway(spec: LoraPoolSpec) -> str:
    return await ModalFlashPool(spec.app_name, "Server").gateway_url_async()


def deploy_pool(spec: LoraPoolSpec, *, config=None, platform=None) -> str:
    pool = ModalFlashPool(spec.app_name, "Server")
    try:
        return pool.gateway_url()
    except Exception as exc:
        if not isinstance(exc, modal.exception.NotFoundError):
            raise
    if config is None:
        saved = pool_config(spec.definition_id)
        if saved is None:
            raise ValueError(f"missing deployment config: {spec.definition_id}")
        return provision_pool(saved, spec, platform or platform_from_env())
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("modal CLI is unavailable")

    platform = platform or config.recipe.platform
    recipe_env = {
        POOL_CONFIG_ENV: config.model_dump_json(),
        PLATFORM_ENV: json.dumps(platform),
    }
    command = [
        modal_cli,
        "deploy",
        "-m",
        "spindle.providers.modal.deployment_pool_app",
        "--name",
        spec.app_name,
    ]
    environment = platform["modal"]["environment"]
    if environment:
        command.extend(["--env", environment])
    subprocess.run(command, env={**os.environ, **spec.env(), **recipe_env}, check=True)
    return pool.gateway_url()


def stop_pool(spec: LoraPoolSpec) -> None:
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("modal CLI is unavailable")
    command = [modal_cli, "app", "stop", "-y", spec.app_name]
    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment:
        command.extend(["--env", environment])
    result = subprocess.run(command, capture_output=True, text=True)
    already_stopped = result.returncode == 1 and any(
        line.strip().startswith("App is already stopped.")
        for line in (result.stdout + "\n" + result.stderr).splitlines()
    )
    if not already_stopped:
        result.check_returncode()
