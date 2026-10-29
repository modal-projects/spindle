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
SGLANG_REVISION = "a3e0e1725406b9330bd062520d4f99d0aa73ee4f"

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
        # Bundle the fork's radix-tree binding so samplers load it in never mode.
        "python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; "
        'assert hasattr(bindings.RustUnifiedTreeCoreBinding, "routing_snapshot_delta"); '
        "import pathlib, shutil; "
        'shutil.copy(bindings.__file__, "/sgl-workspace/sglang/python/sglang/srt/mem_cache/rust_tree_core/" '
        "+ pathlib.Path(bindings.__file__).name)'"
        " && rm -rf /root/.cache/sglang/rust_extensions",
        "SGLANG_RUST_BUILD_MODE=never python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; "
        'assert hasattr(bindings.RustUnifiedTreeCoreBinding, "routing_snapshot_delta")\'',
    )
    .env({"SGLANG_RUST_BUILD_MODE": "never"})
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            # SGLang applies split-K only to the BF16 rank-32 CSGMV shapes it
            # measured, and never in deterministic mode.
            "SGLANG_ENABLE_LORA_CSGMV_SPLIT_K": "1",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
