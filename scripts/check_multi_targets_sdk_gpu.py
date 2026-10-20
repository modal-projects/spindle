"""Run real model training through the Tinker SDK and all Spindle HTTP hops.

uv run --no-sync modal run scripts/check_multi_targets_sdk_gpu.py
"""

import subprocess
from pathlib import Path

import modal

from spindle.providers.modal.miles_image import image

root = Path(__file__).resolve().parents[1]
app = modal.App("spindle-multi-target-sdk-gpu-check")


@app.function(
    image=image.add_local_file(
        root / "tests/gpu/multi_targets_sdk_worker.py",
        "/root/multi_targets_sdk_worker.py",
    ).add_local_file(root / "tests/support.py", "/root/gpu_test_support.py"),
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=1800,
)
def check():
    subprocess.run(
        ["python", "/root/multi_targets_sdk_worker.py"], check=True, timeout=1700
    )


@app.local_entrypoint()
def main():
    check.remote()
