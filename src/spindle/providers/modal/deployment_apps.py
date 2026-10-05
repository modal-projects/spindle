"""Modal app builders shared by all Python-configured model deployments.

Trainers and inference provisioners are independently deployed apps. Pools are
created on demand with the existing LoRA/FFT pool lifecycle.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time

import modal
import modal.experimental
from modal.config import config

from spindle.deployments import DeploymentConfig
from spindle.inference.serving import (
    start_fft_sidecar,
    start_lora_sidecar,
    supervise_children,
    terminate,
    wait_http,
)

from .deployment import trainer_deployment_env
from .fft_pool import FFTPoolSpec
from .fft_pool import deploy_pool as deploy_fft
from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .kernel_cache import KERNEL_CACHE_ENV, KERNEL_CACHE_ROOT, kernel_cache_volume
from .kv import shared_kv
from .lora_pool import LoraPoolSpec
from .lora_pool import deploy_pool as deploy_lora
from .megatron_image import image as megatron_image
from .miles_image import image as miles_image
from .ray_cluster import start_trainer_cluster
from .rollout_image import image as rollout_image
from .serve import run_engine_with_backend


def image_for(backend, image_reference=None):
    if not modal.is_local():
        return modal.Image.debian_slim()
    if image_reference is not None:
        module, attribute = image_reference.split(":", 1)
        return getattr(importlib.import_module(module), attribute)
    return {"miles": miles_image, "megatron": megatron_image, "sglang": rollout_image}[
        backend
    ]


def volumes_for(platform):
    storage = platform["storage"]
    return {
        KERNEL_CACHE_ROOT: kernel_cache_volume,
        "/assets": modal.Volume.from_name(storage["assets"], create_if_missing=True),
        "/checkpoints": modal.Volume.from_name(
            storage["checkpoints"], create_if_missing=True, version=2
        ),
        "/bulletin": modal.Volume.from_name(
            storage["bulletin"], create_if_missing=True, version=2
        ),
    }


def secrets_for(platform, *, training=False):
    names = platform["secrets"]
    result = [modal.Secret.from_name(names["api"], required_keys=["TINKER_API_KEY"])]
    if training:
        result.append(
            modal.Secret.from_name(
                names["sampler_proxy"],
                required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
            )
        )
        if names["huggingface"]:
            result.append(modal.Secret.from_name(names["huggingface"]))
    return result


def deployment_env(values):
    """Keep user environment overrides separate from Spindle's deployment wiring."""
    if any(key.startswith("SPINDLE_") for key in values):
        raise ValueError("SPINDLE_ environment variables are managed by Spindle")
    return dict(values)


def build_trainer_app(deployment: DeploymentConfig, platform=None, *, image=None):
    if platform is None:
        platform = deployment.recipe.platform
    recipe = deployment.recipe
    trainer_name = recipe.name
    app = modal.App(deployment.trainer_app_name)

    env = {
        **trainer_deployment_env(),
        **deployment_env(recipe.trainer_env),
        "SPINDLE_APP_NAME": platform["frontend"],
    }

    def trainer(instance_id: str, config_json: str, frontend_app_id: str):
        saved = DeploymentConfig.model_validate_json(config_json)
        if saved.name != trainer_name:
            raise ValueError("trainer settings do not match the deployed app")
        os.environ["SPINDLE_FRONTEND_APP_ID"] = frontend_app_id
        run_trainer(saved, instance_id, platform)

    if recipe.trainer_nodes > 1:
        trainer = modal.experimental.clustered(recipe.trainer_nodes, rdma=True)(trainer)
    trainer = app.function(
        name="trainer",
        serialized=True,
        image=(
            image
            if image is not None
            else image_for(recipe.backend, recipe.trainer_image)
        ),
        gpu=f"{recipe.trainer_gpu}:{recipe.trainer_gpus_per_node}",
        region=platform["modal"]["region"],
        cpu=recipe.trainer_cpu,
        memory=recipe.trainer_memory_mib,
        timeout=recipe.trainer_timeout_s,
        max_containers=None,
        min_containers=0,
        single_use_containers=True,
        volumes=volumes_for(platform),
        secrets=secrets_for(platform, training=True),
        env=env,
        experimental_options={"efa_enabled": True} if recipe.trainer_nodes > 1 else {},
    )(trainer)

    return app, trainer


def run_trainer(deployment, instance_id, platform=None):
    if platform is None:
        platform = deployment.recipe.platform
    recipe = deployment.recipe
    settings = deployment.trainer_settings
    assets = volumes_for(platform)["/assets"]
    if recipe.trainer_nodes > 1:
        ray_address = start_trainer_cluster(
            recipe.trainer_nodes,
            before_head=assets.reload,
            before_worker_join=assets.reload,
        )
        if ray_address is None:
            return
    else:
        assets.reload()
        ray_address = None
    env = {
        **KERNEL_CACHE_ENV,
        **deployment_env(recipe.trainer_env),
        "SPINDLE_APP_NAME": platform["frontend"],
        "SPINDLE_BACKEND_CONFIG": json.dumps(settings),
        "SPINDLE_BASE_MODEL": recipe.model,
        "SPINDLE_DEFINITION_ID": deployment.definition_id,
        "SPINDLE_CHECKPOINT_VOLUME": platform["storage"]["checkpoints"],
        "SPINDLE_BULLETIN_ROOT": "/bulletin",
        "SPINDLE_BULLETIN_VOLUME": platform["storage"]["bulletin"],
        "SPINDLE_DEFINITION_REVISION": deployment.definition_id,
    }
    if ray_address is not None:
        env["SPINDLE_RAY_ADDRESS"] = ray_address
    executor = (
        "spindle.backends.miles_lora:build_executor"
        if recipe.backend == "miles"
        else "spindle.backends.megatron_fft:build_executor"
    )

    async def failed(error):
        await shared_kv().put(
            f"deployment_failure:{deployment.definition_id}",
            {"error": str(error), "instance_id": instance_id, "failed_at": time.time()},
        )

    run_engine_with_backend(
        shared_kv(),
        executor,
        definition_id=deployment.definition_id,
        revision=config["image_id"],
        instance_id=instance_id,
        backend_env=env,
        nproc=1 if recipe.backend == "miles" else recipe.trainer_gpus_per_node,
        max_models=recipe.trainer_max_clients_per_instance,
        sampler_persistence_concurrency=recipe.sampler_persistence_concurrency,
        on_startup_error=failed,
    )


def build_rollout_app(deployment, pool, platform=None, *, image=None):
    """Create one frozen-base LoRA pool or one FFT latest/pinned/base pool."""
    if platform is None:
        platform = deployment.recipe.platform
    recipe = deployment.recipe
    lora = recipe.parameterization == "lora"
    if pool.definition_id != deployment.definition_id:
        raise ValueError("pool does not match deployment")
    app = modal.App(pool.app_name)
    options = deployment.inference_settings
    minimum = recipe.inference_min_replicas
    maximum = recipe.inference_max_replicas
    window = recipe.inference_scaledown_window_s
    if isinstance(pool, FFTPoolSpec):
        minimum = minimum if pool.min_containers is None else pool.min_containers
        maximum = maximum if pool.max_containers is None else pool.max_containers
        window = window if pool.scaledown_window is None else pool.scaledown_window

    @app.server(
        name="Server",
        serialized=True,
        image=(
            image if image is not None else image_for("sglang", recipe.inference_image)
        ),
        gpu=f"{recipe.inference_gpu}:{recipe.inference_gpus_per_node}",
        cpu=recipe.inference_cpu,
        memory=recipe.inference_memory_mib,
        volumes=volumes_for(platform),
        secrets=secrets_for(platform),
        env=deployment_env(recipe.inference_env),
        min_containers=minimum,
        max_containers=maximum,
        target_concurrency=recipe.inference_target_concurrency,
        scaledown_window=window,
        startup_timeout=recipe.inference_startup_timeout_s,
        exit_grace_period=300,
        port=8000,
        routing_region=platform["modal"]["region"],
        compute_region=platform["modal"]["region"],
    )
    class Server:
        @modal.enter()
        def start(self):
            self.sidecar = None
            self.sglang = None

            self.sglang = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "spindle.inference.sglang",
                    deployment.asset_path,
                    json.dumps(options),
                ],
                start_new_session=True,
            )
            try:
                wait_http(
                    "http://127.0.0.1:8001/health",
                    self.sglang,
                    recipe.inference_startup_timeout_s,
                )
                kwargs = dict(
                    port=8000,
                    sglang_port=8001,
                    bulletin_root="/bulletin",
                    bulletin_volume=platform["storage"]["bulletin"],
                )
                self.sidecar = (
                    start_lora_sidecar(**kwargs)
                    if lora
                    else start_fft_sidecar(
                        **kwargs,
                        model_path=deployment.asset_path,
                        run_id=pool.model_id,
                        pinned_version=None if pool.latest else pool.version,
                    )
                )
                self.supervisor = supervise_children(self.sglang, self.sidecar)
                wait_http(
                    "http://127.0.0.1:8000/health",
                    self.sidecar,
                    recipe.inference_startup_timeout_s,
                )
            except BaseException:
                terminate(self.sidecar)
                terminate(self.sglang)
                raise

        @modal.exit()
        def stop(self):
            terminate(self.sidecar)
            terminate(self.sglang)

    return app, Server


def build_inference_app(deployment, platform=None, *, image=None):
    """Deploy the code that provisions this configuration's inference pools."""
    if platform is None:
        platform = deployment.recipe.platform

    if image is None:
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("git")
            .pip_install(
                *CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "huggingface-hub"
            )
            .add_local_python_source("spindle", copy=True, ignore=ignore_config_source)
        )
    app = modal.App(deployment.inference_app_name)
    inference_name = deployment.name

    @app.function(name="provision", image=image, serialized=True, timeout=1800)
    def provision(config_json: str, pool_data: dict):
        saved = DeploymentConfig.model_validate_json(config_json)
        if saved.name != inference_name:
            raise ValueError("inference settings do not match the deployed app")
        if pool_data["definition_id"] != saved.definition_id:
            raise ValueError("pool definition does not match deployment")
        if saved.parameterization == "lora":
            return deploy_lora(
                LoraPoolSpec.from_dict(pool_data), config=saved, platform=platform
            )
        return deploy_fft(
            FFTPoolSpec.from_dict(pool_data), config=saved, platform=platform
        )

    return app, provision
