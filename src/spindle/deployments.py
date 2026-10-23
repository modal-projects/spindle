"""Python deployment configs: the recipe plus backend settings used to deploy it."""

from __future__ import annotations

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
INFERENCE_GPU_OVERRIDABLE = {"model_weights", "weights_revision", "sglang_cfg", "inference_env"}
# Written into each downloaded weights directory once it is complete.
WEIGHTS_MARKER = ".spindle-weights-revision"
# The frontend routes every sampler request with the base deployment's values.
SHARED_SERVING_SETTINGS = ("tp_size", "dp_size", "enable_dp_attention", "skip_tokenizer_init")


def gpu_family(gpu: str) -> str:
    """The GPU a Modal type names: A100 for "A100-80GB", H100 for "H100!"."""
    return gpu.rstrip("!+").split("-")[0].upper()


def _jsonable(value):
    return json.loads(json.dumps(value))


class DeploymentConfig(BaseModel):
    """Recipe plus the backend settings used to deploy one model."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    recipe: BaseConfig
    trainer_settings: dict
    inference_settings: dict
    # Configs saved before versioning have no version and report 0.1.0.
    spindle_version: str = "0.1.0"

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
        deployment = cls(
            recipe=pinned,
            trainer_settings=_jsonable(trainer_settings),
            inference_settings=_jsonable(inference_settings),
            spindle_version=__version__,
        )
        deployment._check_inference_gpu_overrides()
        return deployment

    def _check_inference_gpu_overrides(self) -> None:
        gpus = self.recipe.inference_gpu
        gpus = [gpus] if isinstance(gpus, str) else gpus
        for gpu, overrides in self.recipe.inference_gpu_overrides.items():
            if gpu not in gpus:
                raise ValueError(f"inference_gpu_overrides: {gpu} is not in inference_gpu")
            keys = {path.split(".")[0] for path in overrides}
            if keys - INFERENCE_GPU_OVERRIDABLE:
                unknown = sorted(keys - INFERENCE_GPU_OVERRIDABLE)
                raise ValueError(f"inference_gpu_overrides cannot set {unknown}")
            weights = keys & {"model_weights", "weights_revision"}
            if weights and self.parameterization != "lora":
                raise ValueError("only LoRA samplers can serve other weights")
            serving = self._with_overrides(overrides)
            for key in SHARED_SERVING_SETTINGS:
                if serving.inference_settings.get(key) != self.inference_settings.get(key):
                    raise ValueError(f"inference_gpu_overrides cannot change {key}")
            if not all(isinstance(v, str) for v in serving.recipe.inference_env.values()):
                raise ValueError("inference_env values must be strings")
        self.weight_sets()

    def _with_overrides(self, overrides: dict) -> DeploymentConfig:
        values = {**vars(self.recipe), "inference_gpu_overrides": {}}
        if "model_weights" in overrides and "weights_revision" not in overrides:
            values["weights_revision"] = ""
        return DeploymentConfig.create(BaseConfig(**values, overrides=overrides))

    def for_inference_gpu(self, gpu_name: str) -> DeploymentConfig:
        """The deployment a sampler on the GPU named `gpu_name` serves."""
        words = re.split(r"[\s-]+", gpu_name.upper())
        for gpu, overrides in self.recipe.inference_gpu_overrides.items():
            if gpu_family(gpu) in words:
                return self._with_overrides(overrides)
        return self

    def weight_sets(self) -> list[tuple[str, str]]:
        """(repo, revision) of the base weights, then of any sampler-only weights."""
        sets = [(self.weights_repo, self.recipe.weights_revision)]
        for overrides in self.recipe.inference_gpu_overrides.values():
            serving = self._with_overrides(overrides)
            weights = (serving.weights_repo, serving.recipe.weights_revision)
            if weights not in sets:
                sets.append(weights)
        if len({repo for repo, _ in sets}) != len(sets):
            raise ValueError("weights for another GPU must come from a different repo")
        return sets

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
            recipe.inference_gpu_overrides,
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
