import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.21"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "spindle-sglang-v0.5.21"

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/spindle-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/spindle-sglang-overlay"
        " && git -C /tmp/spindle-sglang-overlay checkout 12256cadc7445f06402e8058d03e8802cb479476",
        "cp -a /tmp/spindle-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && cp -a /tmp/spindle-sglang-overlay/rust /sgl-workspace/sglang/"
        " && rm -rf /tmp/spindle-sglang-overlay",
        "python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; "
        'assert hasattr(bindings.RustUnifiedTreeCoreBinding, "routing_snapshot_delta")\'',
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .env(
        {
            "SGLANG_RUST_BUILD_MODE": "auto",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
