import json
import os
import re
import subprocess
from importlib.metadata import distribution
from pathlib import Path

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    ignore_config_source,
)

# The release tag pins Miles together with the Megatron-LM and Megatron-Bridge
# revisions its image was built and tested with. Update all three by moving to a
# newer release tag, digest, and commit together, then refresh the trainer app.
MILES_RELEASE = "v0.1.1"
BASE_IMAGE = (
    f"radixark/miles:{MILES_RELEASE}"
    "@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf"
)
MILES_RELEASE_COMMIT = "2806267d060d51b1d3b62f85a1f9b145047aeef9"
MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles"
MILES_COMMIT = "fdd808c008175d4895fd8c57c95ef4fa5bb6a2a0"
MILES_PATH = "/root/miles"
MEGATRON_PATH = "/root/Megatron-LM"


def _head(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def _check_release(miles_commit: str) -> None:
    lock = json.loads(Path(MILES_PATH, "release-lock.json").read_text())
    dockerfile = Path(MILES_PATH, "docker", "Dockerfile").read_text()
    bridge = re.search(r"Megatron-Bridge[.]git@([0-9a-f]{40})", dockerfile)[1]
    installed = json.loads(distribution("megatron-bridge").read_text("direct_url.json"))
    assert _head(MILES_PATH) == miles_commit, (
        "Miles base image differs from the release commit"
    )
    assert _head(MEGATRON_PATH) == lock["megatron_commit"], (
        "Megatron-LM differs from release-lock.json"
    )
    assert installed["vcs_info"]["commit_id"] == bridge, (
        "Megatron-Bridge differs from the Miles Dockerfile"
    )
    print("megatron-lm", lock["megatron_commit"], "megatron-bridge", bridge)


BRIDGE_IMPORT_CACHE_PATCH = (
    "--- a/megatron/bridge/models/conversion/model_bridge.py\n"
    "+++ b/megatron/bridge/models/conversion/model_bridge.py\n"
    "@@ -21,6 +21,7 @@\n"
    " import math\n"
    " import re\n"
    " import warnings\n"
    "+from collections import Counter\n"
    " from dataclasses import dataclass, field, fields, is_dataclass\n"
    " from pathlib import Path\n"
    " from typing import (\n"
    "@@ -1470,6 +1471,11 @@\n"
    "         self.unquantized_state_dict = None\n"
    " \n"
    "         _hf_import_cache: Dict[str, torch.Tensor] = {}\n"
    "+        remaining_grouped_uses = Counter(\n"
    "+            str(task.mapping.hf_param)\n"
    "+            for task in hf_to_megatron_tasks\n"
    '+            if task.megatron_module is not None and getattr(task.mapping, "is_grouped_export", False)\n'
    "+        )\n"
    "         for task in self._with_progress_tracking(hf_to_megatron_tasks, description):\n"
    "             # None means megatron module not on current rank, skip if this task is not going to happen\n"
    "             if task.megatron_module is None:\n"
    "@@ -1483,6 +1489,12 @@\n"
    "                 hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)\n"
    "                 if is_grouped:\n"
    "                     _hf_import_cache[hf_param_key] = hf_weights\n"
    "+            if is_grouped:\n"
    "+                remaining_grouped_uses[hf_param_key] -= 1\n"
    "+                if remaining_grouped_uses[hf_param_key] == 0:\n"
    "+                    # The local reference keeps this tensor alive through conversion.\n"
    "+                    # Retaining earlier layers would cache a full expanded model per rank.\n"
    "+                    del _hf_import_cache[hf_param_key]\n"
    " \n"
    "             # 2) Delegate conversion & distribution to the bridge\n"
    "             converted_weights = self._convert_loaded_hf_weight(task, hf_weights)\n"
)

# The release has the DSA backend hooks but omits the TileLang implementation.
# Upstream #5049 adds the kernels and their packed/context-parallel tests.
# Fetching the commit by hash pins its exact contents, unlike a generated .diff.
DSA_KERNEL_COMMIT = "4f657171b150510cafc48f429673fb1b2c4c098d"
DSA_KERNEL_INSTALL = (
    f"git -C {MEGATRON_PATH} fetch --depth=2 --no-tags"
    f" https://github.com/NVIDIA/Megatron-LM.git {DSA_KERNEL_COMMIT}"
    f" && git -C {MEGATRON_PATH} diff --binary {DSA_KERNEL_COMMIT}^ {DSA_KERNEL_COMMIT}"
    " > /tmp/dsa-kernels.diff"
    f" && git -C {MEGATRON_PATH} apply /tmp/dsa-kernels.diff"
    " && rm /tmp/dsa-kernels.diff"
)


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    .pip_install(
        *MEGATRON_RUNTIME_PACKAGES,
        "xxhash==3.7.1",
        "transformers==5.12.1",
        "pyyaml>=6.0.2",
        "opentelemetry-exporter-otlp==1.43.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.43.0",
        "opentelemetry-exporter-otlp-proto-http==1.43.0",
    )
    .run_function(_check_release, kwargs={"miles_commit": MILES_RELEASE_COMMIT})
    .run_commands(
        f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
        f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
        "python - <<'PY'\n"
        "from importlib.metadata import distribution\n"
        "import subprocess\n"
        "root = distribution('megatron-bridge').locate_file('')\n"
        "subprocess.run(['patch', '-p1', '-d', str(root)], input="
        + repr(BRIDGE_IMPORT_CACHE_PATCH)
        + ", text=True, check=True)\nPY\n",
    )
    .run_commands(
        "pip install --no-deps 'peft>=0.18.1'",
        MEGATRON_RUNTIME_CHECK,
        'python -c "from miles.ray.train.group import TrainerController as T; '
        "assert all(hasattr(T, name) for name in "
        "('load_slot', 'unload_slot', 'forward_backward', "
        "'forward_only', 'optim_step', 'save_slot', 'export_slot'))\"",
    )
    .run_commands(DSA_KERNEL_INSTALL)
    .add_local_python_source("spindle", ignore=ignore_config_source)
)

# Local Miles checkout mounted over the image's Miles package, for testing unreleased Miles changes.
MILES_TREE_ENV = "SPINDLE_MILES_TREE"


def mount_local_miles_tree(base: modal.Image, tree: str) -> modal.Image:
    """Returns ``base`` with ``<tree>/miles`` mounted over the installed Miles package.

    Raises ValueError when ``tree`` is not a Miles checkout.
    """
    package = Path(tree, "miles")
    if not (package / "__init__.py").is_file():
        raise ValueError(f"{MILES_TREE_ENV}={tree} is not a Miles checkout")
    print(
        f"spindle: mounting local Miles tree {package} over {MILES_PATH}/miles",
        flush=True,
    )
    return base.add_local_dir(package, f"{MILES_PATH}/miles")


if os.environ.get(MILES_TREE_ENV):
    image = mount_local_miles_tree(image, os.environ[MILES_TREE_ENV])
