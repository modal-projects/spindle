"""GLM-5.3 Flash dependencies, isolated from the default trainer and sampler."""

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .miles_image import MILES_COMMIT

BASE_IMAGE = "radixark/miles:dev-202610021926"
MEGATRON_REVISION = "e2c4645f253227daf52818bf12767c284757d2a0"
BRIDGE_REVISION = "8cd3466d14d2337c8492827b3712482c2b3e4866"
GLM_BRIDGE_REVISION = "6527b18e8bb0db994a267e6dfd4db7dafc669df9"
SGLANG_REVISION = "efb62ce269b499123e2d1c89005ee4cea8c31098"

# PR 35's model package depends on its earlier GLM5 TileLang provider. Copy only
# those packages onto a recent Bridge revision for Transformers 5.16 support.
# The GLM5 package supplies its TileLang attention implementation; its older
# model registration is omitted because the base revision registers GLM5 itself.
trainer_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "git -C /root/miles fetch --depth 1 https://github.com/radixark/miles.git "
        f"{MILES_COMMIT} && git -C /root/miles checkout --detach FETCH_HEAD",
        "git -C /root/Megatron-LM fetch --depth 1 https://github.com/radixark/Megatron-LM.git "
        f"{MEGATRON_REVISION} && git -C /root/Megatron-LM checkout --detach FETCH_HEAD",
        "git clone --filter=blob:none https://github.com/radixark/Megatron-Bridge.git /opt/glm-bridge"
        f" && git -C /opt/glm-bridge checkout --detach {BRIDGE_REVISION}"
        f" && git -C /opt/glm-bridge fetch --depth 1 origin {GLM_BRIDGE_REVISION}"
        " && git -C /opt/glm-bridge checkout FETCH_HEAD --"
        " src/megatron/bridge/models/glm5 src/megatron/bridge/models/glm5_next"
        " && truncate -s 0 /opt/glm-bridge/src/megatron/bridge/models/glm5/__init__.py"
        " && echo 'from megatron.bridge.models.glm5_next import Glm5NextBridge'"
        " >> /opt/glm-bridge/src/megatron/bridge/models/__init__.py",
        "pip uninstall -y megatron-bridge megatron_bridge",
        "pip install --no-build-isolation --no-deps -e /root/Megatron-LM -e /root/miles -e /opt/glm-bridge",
    )
    .pip_install(
        *CORE_PACKAGES,
        *MEGATRON_RUNTIME_PACKAGES,
        STITCH_PACKAGE,
        "transformers==5.16.0",
    )
    .env({"SPINDLE_MILES_COMMIT": MILES_COMMIT, "CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    .add_local_python_source("spindle", ignore=ignore_config_source)
)

inference_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none https://github.com/sgl-project/sglang.git /opt/glm-sglang"
        f" && git -C /opt/glm-sglang checkout --detach {SGLANG_REVISION}"
        " && pip install --no-build-isolation --no-deps -e /opt/glm-sglang/python",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "transformers==5.16.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "SGLANG_DISABLE_CUDNN_CHECK": "1"})
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
