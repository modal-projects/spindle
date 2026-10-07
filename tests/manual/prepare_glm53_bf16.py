"""Prepare the full BF16 sampling checkpoint before allocating an RL deployment.

PYTHONPATH=src modal run --detach tests/manual/prepare_glm53_bf16.py
"""

import subprocess
import sys
from pathlib import Path

import modal

from spindle.providers.modal.glm53_image import trainer_image

app = modal.App("spindle-glm53-bf16-prepare")
artifacts = modal.Volume.from_name("spindle-glm53-pr26-full-validation")


@app.function(
    image=trainer_image,
    gpu=["H100", "H200", "A100-80GB", "L40S"],
    cpu=16,
    memory=131072,
    timeout=7200,
    max_containers=4,
    region="us",
    volumes={"/validation": artifacts},
)
def prepare(program: str, partition: int = -1):
    artifacts.reload()
    path = Path("/tmp/glm53-bf16.py")
    path.write_text(program)
    command = [sys.executable, "-u", str(path)]
    if partition >= 0:
        command.append(str(partition))
    subprocess.run(command, check=True)


@app.local_entrypoint()
def main():
    program = Path(__file__).with_name("glm53_bf16.py").read_text()
    calls = [prepare.spawn(program, partition) for partition in range(4)]
    for call in calls:
        call.get()
    prepare.remote(program)
