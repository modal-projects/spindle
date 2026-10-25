import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.22@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "spindle-sglang-v0.5.22"

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/spindle-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/spindle-sglang-overlay",
        "cp -a /tmp/spindle-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /tmp/spindle-sglang-overlay",
        'python -c "import sglang.srt.mem_cache.rust_tree_core.mem_cache"',
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
