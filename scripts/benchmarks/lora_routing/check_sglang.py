"""Run native adapter lifetime tests in the rollout image without allocating GPUs."""

from pathlib import Path
import subprocess
import sys

import modal

from spindle.providers.modal.rollout_image import image

app = modal.App("spindle-lora-routing-tests")
image = image.add_local_file(
    Path(__file__).resolve().parents[3]
    / "tests/inference/test_sglang_lora_lifetime.py",
    "/tmp/test_sglang_lora_lifetime.py",
)


@app.function(image=image, cpu=2, memory=16384, timeout=300)
def check():
    # The rollout image already has runtime mounts; install the test runner in
    # this disposable CPU container instead of changing the production image.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pytest"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "/tmp/test_sglang_lora_lifetime.py"],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        code, output = check.remote()
        print(output, flush=True)
    raise SystemExit(code)
