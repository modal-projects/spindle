"""Python deployment configs: the recipe plus backend settings used to deploy it."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import sys
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from spindle import __version__
from spindle.backends.deployment import resolve_backend_settings
from spindle.configuration import BaseConfig

# Modal's smallest ephemeral disk request.
MIN_EPHEMERAL_DISK_MIB = 524_288
MODAL_APP_NAME_MAX = 63
RUN_SEPARATOR = ".run-"


def _jsonable(value):
    return json.loads(json.dumps(value))


def run_key(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()[:12]


def split_run_definition(definition_id: str) -> tuple[str, str | None]:
    """Return the recipe name and run key of a per-run definition ID."""
    name, separator, run = definition_id.partition(RUN_SEPARATOR)
    return name, (run if separator else None)


def modal_app_name(prefix: str, definition_id: str) -> str:
    name = f"{prefix}{definition_id}"
    if len(name) <= MODAL_APP_NAME_MAX:
        return name
    digest = hashlib.sha256(definition_id.encode()).hexdigest()[:10]
    return f"{name[: MODAL_APP_NAME_MAX - len(digest) - 1]}-{digest}"


class DeploymentConfig(BaseModel):
    """Recipe plus the backend settings used to deploy one model."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    recipe: BaseConfig
    trainer_settings: dict
    inference_settings: dict
    # Configs saved before versioning have no version and report 0.1.0.
    spindle_version: str = "0.1.0"
    # Set on a per-run copy of a recipe with per_run_apps enabled.
    run: str | None = None
    run_tags: dict[str, str] = {}

    def for_run(self, run: str, tags: dict[str, str] | None = None):
        return self.model_copy(update={"run": run, "run_tags": dict(tags or {})})

    @field_validator("recipe", mode="before")
    @classmethod
    def restore_recipe(cls, value):
        return BaseConfig(**value) if isinstance(value, dict) else value

    @field_serializer("recipe")
    def serialize_recipe(self, recipe):
        return vars(recipe)

    @classmethod
    def create(cls, recipe: BaseConfig) -> DeploymentConfig:
        """Copy the recipe and attach backend settings."""
        pinned = BaseConfig(**_jsonable(vars(recipe)))
        for field in ("trainer_disk_mib", "inference_disk_mib"):
            disk = getattr(pinned, field)
            if disk and disk < MIN_EPHEMERAL_DISK_MIB:
                raise ValueError(
                    f"{field} must be 0 or at least {MIN_EPHEMERAL_DISK_MIB}"
                )
        trainer_settings, inference_settings = resolve_backend_settings(
            pinned, f"/assets/{pinned.model_weights or pinned.model}"
        )
        return cls(
            recipe=pinned,
            trainer_settings=_jsonable(trainer_settings),
            inference_settings=_jsonable(inference_settings),
            spindle_version=__version__,
        )

    @property
    def name(self) -> str:
        return self.recipe.name

    @property
    def model(self) -> str:
        return self.recipe.model

    @property
    def parameterization(self) -> str:
        return self.recipe.parameterization

    @property
    def max_context_length(self) -> int:
        return self.recipe.max_context_length

    @property
    def definition_id(self) -> str:
        if self.run is None:
            return self.recipe.name
        return f"{self.recipe.name}{RUN_SEPARATOR}{self.run}"

    @property
    def app_tags(self) -> dict[str, str]:
        return {**self.recipe.tags, **self.run_tags}

    @property
    def trainer_app_name(self) -> str:
        return modal_app_name("spindle-trainer-", self.definition_id)

    @property
    def inference_app_name(self) -> str:
        return f"spindle-inference-{self.recipe.name}"

    @property
    def weights_repo(self) -> str:
        return self.recipe.model_weights or self.recipe.model

    @property
    def asset_path(self) -> str:
        return f"/assets/{self.weights_repo}"

    @property
    def rollout_tensor_parallel_size(self) -> int:
        serving = self.inference_settings
        return serving.get("tp_size", self.recipe.inference_gpus_per_node) // (
            serving.get("dp_size", 1) if serving.get("enable_dp_attention") else 1
        )

    def trainer_identity(self) -> tuple:
        recipe = self.recipe
        return (
            recipe.name,
            recipe.model,
            recipe.model_weights,
            recipe.weights_revision,
            recipe.parameterization,
            recipe.max_context_length,
            recipe.backend,
            recipe.miles_cfg,
            recipe.megatron_cfg,
            recipe.sampler_persistence_concurrency,
            recipe.trainer_gpu,
            recipe.trainer_gpus_per_node,
            recipe.trainer_nodes,
            recipe.trainer_cpu,
            recipe.trainer_memory_mib,
            recipe.trainer_disk_mib,
            recipe.trainer_max_instances,
            recipe.trainer_max_clients_per_instance,
            recipe.trainer_max_clients_per_batch,
            recipe.trainer_timeout_s,
            recipe.trainer_env,
            recipe.trainer_image,
            recipe.tags,
            self.trainer_settings,
            self.spindle_version,
        )

    def inference_identity(self) -> tuple:
        recipe = self.recipe
        return (
            recipe.name,
            recipe.model,
            recipe.model_weights,
            recipe.weights_revision,
            recipe.parameterization,
            recipe.max_context_length,
            recipe.sglang_cfg,
            recipe.inference_gpu,
            recipe.inference_gpus_per_node,
            recipe.inference_cpu,
            recipe.inference_memory_mib,
            recipe.inference_disk_mib,
            recipe.inference_min_replicas,
            recipe.inference_max_replicas,
            recipe.inference_target_concurrency,
            recipe.inference_scaledown_window_s,
            recipe.inference_startup_timeout_s,
            recipe.inference_env,
            recipe.tags,
            self.inference_settings,
            self.spindle_version,
        )


def config_path(name: str) -> Path:
    """Locate an installed example config without maintaining a model catalog."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("invalid config name")
    return Path(
        str(files("spindle").joinpath("configs", name.replace("-", "_") + ".py"))
    )


def load(path: str | Path) -> BaseConfig:
    """Execute a Python config file and read its exported config object."""
    path = Path(path).resolve()
    if path.suffix != ".py":
        raise ValueError("deployment configs must be Python .py files")
    original_path = sys.path[:]
    sys.path.insert(0, str(path.parent))
    try:
        namespace = runpy.run_path(str(path))
        config = namespace.get("config")
        if not isinstance(config, BaseConfig):
            raise ValueError(f"{path} must export a BaseConfig instance named config")
        return config
    finally:
        sys.path[:] = original_path


def validate_frontend(recipes: list[BaseConfig]) -> None:
    if not recipes:
        raise ValueError("at least one deployment is required")
    if len({recipe.name for recipe in recipes}) != len(recipes):
        raise ValueError("duplicate deployment name")
    first = recipes[0]
    for recipe in recipes:
        if RUN_SEPARATOR in recipe.name:
            raise ValueError(f"deployment names cannot contain {RUN_SEPARATOR!r}")
        if recipe.per_run_apps and recipe.parameterization != "lora":
            raise ValueError("per_run_apps requires a LoRA deployment")
        if recipe.platform != first.platform:
            raise ValueError("deployments on one frontend must share platform settings")
        if (
            recipe.session_idle_timeout_s != first.session_idle_timeout_s
            or recipe.pool_idle_timeout_s != first.pool_idle_timeout_s
            or recipe.sweep_interval_s != first.sweep_interval_s
        ):
            raise ValueError(
                "deployments on one frontend must share lifecycle settings"
            )
