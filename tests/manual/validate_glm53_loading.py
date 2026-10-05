"""Validate EP-local and asynchronous loading in the actual GLM SGLang image.

PYTHONPATH=src modal run --detach tests/manual/validate_glm53_loading.py
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal.glm53_image import inference_image
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)

app = modal.App("spindle-glm53-loading-validation")
artifacts = modal.Volume.from_name("spindle-glm53-pr26-validation")


@app.function(
    image=inference_image,
    cpu=4,
    memory=16384,
    timeout=600,
    volumes={
        "/validation": modal.Volume.from_name("spindle-glm53-pr26-rl-bulletin", version=2)
    },
)
def checks(program: str):
    path = Path("/tmp/glm53-loading-checks.py")
    path.write_text(program)
    subprocess.run([sys.executable, "-u", str(path)], check=True)


@app.function(
    image=inference_image,
    gpu="H200:2",
    cpu=16,
    memory=131072,
    timeout=2400,
    region="us",
    env=KERNEL_CACHE_ENV,
    volumes={"/validation": artifacts, KERNEL_CACHE_ROOT: kernel_cache_volume},
)
def serving(program: str, settings: dict):
    path = Path("/tmp/glm53-loading-serving.py")
    path.write_text(program)
    subprocess.run([sys.executable, "-u", str(path), json.dumps(settings)], check=True)


@app.local_entrypoint()
def main(cpu_only: bool = False):
    root = Path(__file__).parent
    call = checks.spawn((root / "glm53_loading_checks.py").read_text())
    print("CPU loading checks", call.object_id, flush=True)
    call.get()
    if cpu_only:
        return
    config = DeploymentConfig.create(load(config_path("glm53-flash-lora-16k")))
    program = (root / "glm53_sampling.py").read_text()
    settings = {
        **config.inference_settings,
        "validation_parallel_size": 2,
        "validation_measure_loading": True,
    }
    call = serving.spawn(program, settings)
    print("TP2/EP2 serving", call.object_id, flush=True)
    call.get()
