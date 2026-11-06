import json
import re
import subprocess
from importlib.metadata import distribution
from pathlib import Path

import modal

from .bridge_import_cache import BRIDGE_IMPORT_CACHE_PATCH

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
