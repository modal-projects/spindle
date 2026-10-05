import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import modal

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal import deployment_apps
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    KERNEL_CACHE_VOLUME_NAME,
    kernel_cache_volume,
)


def test_kernel_cache_creates_one_v2_volume_without_live_lookup() -> None:
    module_path = (
        Path(__file__).parents[2] / "src/spindle/providers/modal/kernel_cache.py"
    )
    with patch.object(
        modal.Volume,
        "from_name",
        return_value=sentinel.kernel_cache_volume,
    ) as from_name:
        module = runpy.run_path(str(module_path))

    from_name.assert_called_once_with(
        "spindle-kernel-cache",
        create_if_missing=True,
        version=2,
    )
    assert module["KERNEL_CACHE_ROOT"] == "/root/.cache/kernel-cache"
    assert module["KERNEL_CACHE_ENV"] == {
        "TRITON_CACHE_DIR": "/root/.cache/kernel-cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernel-cache/inductor",
    }


def test_configured_trainers_mount_the_shared_kernel_cache() -> None:
    assert KERNEL_CACHE_VOLUME_NAME == "spindle-kernel-cache"
    volumes = deployment_apps.volumes_for(BaseConfig().platform)
    assert volumes[KERNEL_CACHE_ROOT] is kernel_cache_volume
    assert KERNEL_CACHE_ENV["TRITON_CACHE_DIR"].startswith(KERNEL_CACHE_ROOT + "/")
    assert KERNEL_CACHE_ENV["TORCHINDUCTOR_CACHE_DIR"].startswith(
        KERNEL_CACHE_ROOT + "/"
    )


def test_configured_trainers_point_compilers_at_the_kernel_cache() -> None:
    for preset in ("qwen35-9b-lora-16k", "qwen35-4b-fft-64k"):
        deployment = DeploymentConfig.create(load(config_path(preset)))
        with (
            patch.object(deployment_apps, "volumes_for") as volumes,
            patch.object(deployment_apps, "shared_kv"),
            patch.object(deployment_apps, "run_engine_with_backend") as run,
        ):
            deployment_apps.run_trainer(deployment, "test-instance")
        volumes.return_value.__getitem__.return_value.reload.assert_called_once()
        env = run.call_args.kwargs["backend_env"]
        for key, value in KERNEL_CACHE_ENV.items():
            assert env[key] == value
