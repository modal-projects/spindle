"""Run native scoring/backward regression on four temporary H100 GPUs.

    uv run --no-sync modal run scripts/check_multi_targets_gpu.py

Uses the pinned production Miles image and real TP/CP kernels. Synthetic logits
isolate scoring correctness; this does not launch a model-serving deployment.
"""

import json
import subprocess
from pathlib import Path

import modal

from spindle.providers.modal.miles_image import image

app = modal.App("spindle-multi-target-gpu-check")
worker = Path(__file__).resolve().parents[1] / "tests/gpu/multi_targets_worker.py"


@app.function(
    image=image.add_local_file(worker, "/root/multi_targets_worker.py"),
    gpu="H100:4",
    timeout=900,
)
def check():
    results = []
    for tp, cp in ((1, 1), (2, 1), (1, 2), (2, 2)):
        process = subprocess.Popen(
            [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={tp * cp}",
                "/root/multi_targets_worker.py",
                "--tp",
                str(tp),
                "--cp",
                str(cp),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("GPU_RESULT "):
                results.extend(json.loads(line.removeprefix("GPU_RESULT ")))
        if process.wait() != 0:
            raise RuntimeError(f"GPU regression failed for TP={tp}, CP={cp}")
    return results


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=2))
