import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.21"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
# Modal caches this layer by command, so fetching a branch by name would keep
# serving an older head.
SGLANG_COMMIT = "a3e0e1725406b9330bd062520d4f99d0aa73ee4f"

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/spindle-sglang-overlay"
        " && git init -q /tmp/spindle-sglang-overlay"
        f" && git -C /tmp/spindle-sglang-overlay fetch --depth=1 {SGLANG_REPOSITORY} {SGLANG_COMMIT}"
        " && git -C /tmp/spindle-sglang-overlay checkout -q FETCH_HEAD",
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
