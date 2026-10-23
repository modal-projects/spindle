"""Bounded H200 validation. Each invocation stops its GPU when the call ends."""

import json
from pathlib import Path
import subprocess
import sys

import modal

from spindle.providers.modal.torchtitan_image import image

app = modal.App("spindle-torchtitan-benchmark")
benchmark_image = image.add_local_file(
    Path(__file__).with_name("benchmark_torchtitan.py"),
    "/root/benchmark_torchtitan.py",
)


@app.function(image=benchmark_image, gpu="H200", cpu=16, memory=131072, timeout=3000)
def benchmark(
    model: str,
    flavor: str,
    sequence_length: int,
    adapter_counts: str,
    steps: int,
    compile_layers: bool,
):
    command = [
        sys.executable,
        "/root/benchmark_torchtitan.py",
        "--model",
        model,
        "--flavor",
        flavor,
        "--sequence-length",
        str(sequence_length),
        "--adapter-counts",
        adapter_counts,
        "--steps",
        str(steps),
        "--output",
        "/tmp/result.json",
    ]
    if compile_layers:
        command.append("--compile-layers")
    subprocess.run(command, check=True, timeout=2800)
    return json.loads(Path("/tmp/result.json").read_text())


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.8-27B",
    flavor: str = "27B",
    sequence_length: int = 1024,
    adapter_counts: str = "1,2,4,8",
    steps: int = 3,
    compile_layers: bool = False,
    output: str = "torchtitan-benchmark.json",
):
    result = benchmark.remote(
        model, flavor, sequence_length, adapter_counts, steps, compile_layers
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
