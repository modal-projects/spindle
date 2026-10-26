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
# The Python and Rust trees must come from the same revision: the scheduler
# calls methods on the radix-tree binding that this checkout compiles.
SGLANG_REVISION = "b6a84dde1a37e0f14518eb42911bad3ff0b62720"

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    # The base image sets `never`, which loads its bundled binding and ignores
    # the Rust sources copied below.
    .env({"SGLANG_RUST_BUILD_MODE": "auto"})
    .run_commands(
        "rm -rf /tmp/spindle-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/spindle-sglang-overlay"
        f" && git -C /tmp/spindle-sglang-overlay checkout {SGLANG_REVISION}",
        "cp -a /tmp/spindle-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /sgl-workspace/sglang/rust"
        " && cp -a /tmp/spindle-sglang-overlay/rust /sgl-workspace/sglang/rust"
        " && rm -rf /tmp/spindle-sglang-overlay",
        # Compile the radix-tree binding into SGLang's fingerprinted cache now,
        # instead of on every container's first cache construction.
        "python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; "
        'assert hasattr(bindings.RustUnifiedTreeCoreBinding, "routing_snapshot_delta")\'',
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
