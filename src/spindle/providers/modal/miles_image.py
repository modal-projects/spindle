import json
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
MILES_RELEASE = "v0.1.2"
BASE_IMAGE = (
    f"radixark/miles:{MILES_RELEASE}"
    "@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7"
)
MILES_RELEASE_COMMIT = "a22fd824191c64ba7c5d6c531f125406e4ce9cc6"
MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"
MILES_COMMIT = "83e458d967b838333de098defbf5cf672ef4ce28"
MILES_PATH = "/root/miles"
# Megatron-LM is only partially installed as a package; some model
# construction imports resolve from the source tree.
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

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "PYTHONPATH": f"{MEGATRON_PATH}:$PYTHONPATH",
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
        'python -c "from miles.ray.placement_group import create_trainer_handles; '
        "from miles.ray.train.group import TrainerController as T; "
        "assert all(hasattr(T, name) for name in "
        "('load_slot', 'unload_slot', 'forward_backward', "
        "'forward_only', 'optim_step', 'save_slot', 'export_slot'))\"",
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
