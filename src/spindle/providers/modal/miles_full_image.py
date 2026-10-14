"""Trainer image for full-parameter training through Miles.

Full-parameter recipes track a newer Miles release than the multi-LoRA
image: v0.1.2 is the first release with DeepSeek-V4.1 training support.
"""

import subprocess

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    ignore_config_source,
)

MILES_RELEASE = "v0.1.2"
BASE_IMAGE = (
    f"radixark/miles:{MILES_RELEASE}"
    "@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7"
)
MILES_COMMIT = "a22fd824191c64ba7c5d6c531f125406e4ce9cc6"
MILES_PATH = "/root/miles"


def _check_release(miles_commit: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", MILES_PATH, "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == miles_commit, f"Miles is at {head}, not MILES_COMMIT"


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "SPINDLE_MILES_COMMIT": MILES_COMMIT,
        }
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    .pip_install(
        *MEGATRON_RUNTIME_PACKAGES,
        "pyyaml>=6.0.2",
        "opentelemetry-exporter-otlp==1.43.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.43.0",
        "opentelemetry-exporter-otlp-proto-http==1.43.0",
    )
    .run_function(_check_release, kwargs={"miles_commit": MILES_COMMIT})
    .run_commands(
        MEGATRON_RUNTIME_CHECK,
        # GPU-only modules cannot be imported while building; check what loads on CPU.
        'python -c "from miles.ray.train.group import TrainerController; '
        'from megatron.core import dist_checkpointing"',
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
