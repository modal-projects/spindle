"""Opt-in engine image for SGLang merge PR #44354, pinned independently."""

from pathlib import Path

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_MERGE_REVISION = "708d38f143697e25c3e687c20fc0b043930c8416"
base_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20261014-684fea8a")
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none --single-branch --branch codex/llm-lora-merge "
        "https://github.com/kailash109/sglang.git /opt/dlora-sglang && "
        f"git -C /opt/dlora-sglang checkout --detach {SGLANG_MERGE_REVISION}"
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "peft")
    .add_local_file(
        Path(__file__).resolve().parents[2] / "inference/sglang_routing_observer.py",
        "/opt/dlora-observer.py",
        copy=True,
    )
    .run_commands(
        "python -c 'import runpy; from pathlib import Path; "
        'runpy.run_path("/opt/dlora-observer.py")["install"]'
        '(Path("/opt/dlora-sglang/python/sglang"), modern=True)\''
    )
    .env(
        {
            "PYTHONPATH": "/opt/dlora-sglang/python:/opt/dlora-spindle",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }
    )
)
image = base_image.add_local_python_source("spindle", ignore=ignore_config_source)
