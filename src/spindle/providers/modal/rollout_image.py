import modal
from pathlib import Path

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.21"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "spindle-sglang-v0.5.21"

base_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/spindle-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/spindle-sglang-overlay"
        " && git -C /tmp/spindle-sglang-overlay checkout 79e38a5605a576cae7034d9ab70047fe478e26ee",
        "cp -a /tmp/spindle-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && cp -a /tmp/spindle-sglang-overlay/rust /sgl-workspace/sglang/"
        " && rm -rf /tmp/spindle-sglang-overlay",
        'python -c "import sglang.srt.mem_cache.rust_tree_core.mem_cache"',
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .add_local_file(
        Path(__file__).resolve().parents[2] / "inference/sglang_adapter_observer.py",
        "/opt/spindle-adapter-observer.py",
        copy=True,
    )
    .run_commands("python /opt/spindle-adapter-observer.py")
    .add_local_file(
        Path(__file__).resolve().parents[2] / "inference/sglang_routing_observer.py",
        "/opt/rl-measurement/spindle/inference/sglang_routing_observer.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).resolve().parents[2]
        / "inference/install_routing_observation.py",
        "/opt/install-observation.py",
        copy=True,
    )
    .run_commands(
        "python /opt/install-observation.py",
        "SGLANG_RUST_BUILD_MODE=force python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; assert hasattr(bindings.RustUnifiedTreeCoreBinding, \"routing_snapshot\")'",
    )
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
)

# Benchmark-only backport of upstream PR41951 onto the identical pinned fork.
# The original image above remains the no-optimization control.
split_k_base_image = base_image.add_local_file(
    Path(__file__).resolve().parents[2] / "inference/install_split_k.py",
    "/opt/install-split-k.py",
    copy=True,
).run_commands("python /opt/install-split-k.py")

image = base_image.add_local_python_source("spindle", ignore=ignore_config_source)
split_k_image = split_k_base_image.add_local_python_source(
    "spindle", ignore=ignore_config_source
)
