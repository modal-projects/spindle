"""TorchTitan training dependencies, separate from the Megatron runtime."""

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

TORCHTITAN_COMMIT = "58350410517dd62a15026e6816be874d0d5cb548"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.16.0.dev20261023",
        index_url="https://download.pytorch.org/whl/nightly/cu130",
    )
    .pip_install(
        f"torchtitan @ git+https://github.com/pytorch/torchtitan.git@{TORCHTITAN_COMMIT}"
    )
    .pip_install(
        *CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "transformers==5.12.1", "pytest"
    )
    .env(
        {
            "SPINDLE_TORCHTITAN_COMMIT": TORCHTITAN_COMMIT,
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
