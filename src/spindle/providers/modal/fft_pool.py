from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from typing import Any

import httpx
import modal
from stitch.pools.modal_flash import ModalFlashPool, list_flash_containers
from stitch.types import VersionRef

from .deployment_configs import (
    PLATFORM_ENV,
    POOL_CONFIG_ENV,
    platform_from_env,
    pool_config,
    provision_pool,
)

logger = logging.getLogger(__name__)


def proxy_auth_headers() -> dict[str, str]:
    return {
        "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
    }


@dataclass(frozen=True)
class FFTPoolSpec:
    definition_id: str
    model_id: str
    latest: bool
    version: int
    min_containers: int | None = None
    max_containers: int | None = None
    scaledown_window: int | None = None

    @classmethod
    def base(cls, definition_id: str) -> FFTPoolSpec:
        return cls(definition_id, "base", False, 0)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FFTPoolSpec:
        return cls(**{f.name: value[f.name] for f in fields(cls) if f.name in value})

    @property
    def app_name(self) -> str:
        digest = hashlib.sha256(
            f"{self.definition_id}\0{self.model_id}".encode()
        ).hexdigest()[:16]
        suffix = "latest" if self.latest else f"v{self.version}"
        return f"spindle-fft-{digest}-{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def env(self) -> dict[str, str]:
        return {
            "SPINDLE_FFT_POOL_APP_NAME": self.app_name,
            **{
                f"SPINDLE_FFT_POOL_{key.upper()}": str(
                    int(value) if isinstance(value, bool) else value
                )
                for key, value in self.as_dict().items()
                if value is not None
            },
        }


class FFTLatestPool(ModalFlashPool):
    def __init__(self, definition_id: str, model_id: str) -> None:
        super().__init__(
            FFTPoolSpec(definition_id, model_id, True, 0).app_name,
            "Server",
        )

    def discover_replicas(self) -> list[str]:
        try:
            return super().discover_replicas()
        except modal.exception.NotFoundError:
            return []

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        if not replicas:
            return
        upstreams = [
            upstream
            for container in list_flash_containers(self.app_name, self.cls_name)
            if (upstream := _container_upstream(container)) is not None
        ]
        if not upstreams:
            return
        gateway = self.gateway_url()
        with httpx.Client(
            timeout=5.0,
            trust_env=False,
            headers=proxy_auth_headers(),
        ) as client:

            def wake_one(upstream: str) -> None:
                try:
                    client.post(
                        f"{gateway}/wake",
                        headers={"modal-flash-upstream": upstream},
                    ).raise_for_status()
                except Exception as exc:
                    logger.warning(
                        "failed to wake upstream %s for %s: %s",
                        upstream,
                        ref.identity,
                        exc,
                    )

            with ThreadPoolExecutor(max_workers=min(16, len(upstreams))) as workers:
                list(workers.map(wake_one, upstreams))


async def pool_gateway(spec: FFTPoolSpec) -> str:
    return await ModalFlashPool(spec.app_name, "Server").gateway_url_async()


def deploy_pool(spec: FFTPoolSpec, *, config=None, platform=None) -> str:
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
    env = {**os.environ, **spec.env(), **recipe_env}
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
    subprocess.run(command, env=env, check=True)
    return pool.gateway_url()


def stop_pool(spec: FFTPoolSpec) -> None:
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("modal CLI is unavailable")
    command = [modal_cli, "app", "stop", "-y", spec.app_name]
    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment:
        command.extend(["--env", environment])
    result = subprocess.run(command, capture_output=True, text=True)
    # Modal's CLI exits with 1 when the desired stopped state already holds.
    # Keep other failures visible so the registry entry can be retried later.
    already_stopped = result.returncode == 1 and any(
        line.strip().startswith("App is already stopped.")
        for line in (result.stdout + "\n" + result.stderr).splitlines()
    )
    if not already_stopped:
        result.check_returncode()


def _container_upstream(container) -> str | None:
    if isinstance(container, dict):
        host = container.get("host")
        port = container.get("port")
    else:
        host = getattr(container, "host", None)
        port = getattr(container, "port", None)
    if not host:
        return None
    host = str(host).removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"{host}:{port}" if port else host
