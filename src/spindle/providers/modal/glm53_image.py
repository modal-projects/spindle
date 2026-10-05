"""GLM-5.3 Flash dependencies, isolated from the default trainer and sampler."""

import modal

from .glm53_lora_loading_patch import SGLANG_GLM_LORA_LOADING_PATCH
from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .miles_image import MILES_COMMIT
from .rollout_image import SGLANG_LORA_LIFETIME_PATCH

BASE_IMAGE = "radixark/miles:dev-202610021926"
# Official nightly-dev-cu13-20261005-f70e8c68: PyTorch 2.14.1 / sglang-kernel 0.4.9.
INFERENCE_BASE_IMAGE = "lmsysorg/sglang@sha256:f08b3c7f14bdf2581ea776829a4a2d714825b385c189c0ae53590704676918ff"
MEGATRON_REVISION = "fd15ee20a4f03b03680529baf4b8f7eeed64df1d"
BRIDGE_REVISION = "8cd3466d14d2337c8492827b3712482c2b3e4866"
GLM_BRIDGE_REVISION = "6527b18e8bb0db994a267e6dfd4db7dafc669df9"

# New SGLang owns request completion, but still needs atomic lookup + pin.
_REGISTRY_PATCH_HEADER = "--- a/python/sglang/srt/lora/lora_registry.py\n"
SGLANG_REGISTRY_PATCH = (
    _REGISTRY_PATCH_HEADER
    + SGLANG_LORA_LIFETIME_PATCH.split(_REGISTRY_PATCH_HEADER, 1)[1]
)

# KDA and DSA have different projection widths in the same model.
SGLANG_GLM_LORA_DIMENSIONS_PATCH = (
    "--- a/python/sglang/srt/models/glm5_next.py\n"
    "+++ b/python/sglang/srt/models/glm5_next.py\n"
    "@@ -63,6 +63,7 @@\n"
    "     VocabParallelEmbedding,\n"
    "     get_embedding_tp_kwargs,\n"
    " )\n"
    "+from sglang.srt.lora.utils import get_default_hidden_dim\n"
    " from sglang.srt.managers.mm_utils import (\n"
    "     MultiModalityDataPaddingPatternMultimodalTokens,\n"
    "     general_mm_embed_routine,\n"
    "@@ -1205,6 +1206,16 @@\n"
    "     }\n"
    "     fall_back_to_pt_during_load = False\n"
    " \n"
    "+    def get_hidden_dim(self, module_name: str, layer_idx: int):\n"
    '+        if self.config.layer_types[layer_idx] == "linear_attention":\n'
    "+            linear = self.config.linear_attn_config\n"
    '+            width = linear["num_heads"] * linear["head_dim"]\n'
    '+            if module_name == "qkv_proj":\n'
    "+                return self.config.hidden_size, 3 * width\n"
    '+            if module_name == "o_proj":\n'
    "+                return width, self.config.hidden_size\n"
    "+        return get_default_hidden_dim(module_name, self.config, layer_idx)\n"
    "+\n"
    "     def __init__(\n"
    "         self,\n"
    "         config: Glm5NextConfig,\n"
)

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
        "git -C /root/Megatron-LM fetch --depth 1 https://github.com/NVIDIA/Megatron-LM.git "
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
        "opentelemetry-exporter-otlp==1.43.0",
        "tilelang==0.1.12",
    )
    .env(
        {
            "SPINDLE_MILES_COMMIT": MILES_COMMIT,
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONPATH": "/root/Megatron-LM:/root/miles",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)

inference_image = (
    modal.Image.from_registry(INFERENCE_BASE_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "cd /sgl-workspace/sglang && git apply --check - <<'PATCH'\n"
        + SGLANG_REGISTRY_PATCH
        + SGLANG_GLM_LORA_DIMENSIONS_PATCH
        + SGLANG_GLM_LORA_LOADING_PATCH
        + "PATCH\n",
        "cd /sgl-workspace/sglang && git apply - <<'PATCH'\n"
        + SGLANG_REGISTRY_PATCH
        + SGLANG_GLM_LORA_DIMENSIONS_PATCH
        + SGLANG_GLM_LORA_LOADING_PATCH
        + "PATCH\n",
    )
    .pip_install(
        *CORE_PACKAGES,
        STITCH_PACKAGE,
        TINKER_PACKAGE,
        "opentelemetry-exporter-otlp==1.43.0",
        "opentelemetry-exporter-prometheus==0.64b0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
