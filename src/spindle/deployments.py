"""Python deployment configs: the recipe plus backend settings used to deploy it."""

from __future__ import annotations

import json
import re
import runpy
import sys
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from spindle.backends.deployment import resolve_backend_settings
from spindle.configuration import BaseConfig


def _jsonable(value):
    return json.loads(json.dumps(value))


class DeploymentConfig(BaseModel):
    """Recipe plus the backend settings used to deploy one model."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    recipe: BaseConfig
    trainer_settings: dict
    inference_settings: dict

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
        trainer_settings, inference_settings = resolve_backend_settings(
            pinned, f"/assets/{pinned.model}"
        )
        return cls(
            recipe=pinned,
            trainer_settings=_jsonable(trainer_settings),
            inference_settings=_jsonable(inference_settings),
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
        return self.recipe.name

    @property
    def trainer_app_name(self) -> str:
        return f"spindle-trainer-{self.recipe.name}"

    @property
    def inference_app_name(self) -> str:
        return f"spindle-inference-{self.recipe.name}"

    @property
    def asset_path(self) -> str:
        return f"/assets/{self.recipe.model}"

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
            recipe.parameterization,
            recipe.max_context_length,
            recipe.backend,
            recipe.miles_cfg,
            recipe.megatron_cfg,
            recipe.sampler_persistence_concurrency,
            recipe.trainer_image,
            recipe.trainer_gpu,
            recipe.trainer_gpus_per_node,
            recipe.trainer_nodes,
            recipe.trainer_cpu,
            recipe.trainer_memory_mib,
            recipe.trainer_max_instances,
            recipe.trainer_max_clients_per_instance,
            recipe.trainer_timeout_s,
            recipe.trainer_env,
            self.trainer_settings,
        )

    def inference_identity(self) -> tuple:
        recipe = self.recipe
        return (
            recipe.name,
            recipe.model,
            recipe.parameterization,
            recipe.max_context_length,
            recipe.sglang_cfg,
            recipe.inference_image,
            recipe.inference_gpu,
            recipe.inference_gpus_per_node,
            recipe.inference_cpu,
            recipe.inference_memory_mib,
            recipe.inference_min_replicas,
            recipe.inference_max_replicas,
            recipe.inference_target_concurrency,
            recipe.inference_scaledown_window_s,
            recipe.inference_startup_timeout_s,
            recipe.inference_env,
            self.inference_settings,
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
