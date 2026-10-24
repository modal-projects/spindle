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


@app.function(image=benchmark_image, gpu="H200", cpu=16, memory=131072, timeout=1200)
def benchmark(
    model: str,
    flavor: str,
    sequence_length: int,
    adapter_counts: str,
    steps: int,
    compile_layers: bool,
    diagnose: bool,
    max_batch_tokens: int,
    max_batch_rows: int,
    timeout_seconds: int,
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
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--max-batch-rows",
        str(max_batch_rows),
        "--output",
        "/tmp/result.json",
    ]
    if compile_layers:
        command.append("--compile-layers")
    if diagnose:
        command.append("--diagnose")
    subprocess.run(command, check=True, timeout=timeout_seconds - 60)
    return json.loads(Path("/tmp/result.json").read_text())


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.8-27B",
    flavor: str = "27B",
    sequence_length: int = 1024,
    adapter_counts: str = "1,2,4,8",
    steps: int = 3,
    compile_layers: bool = False,
    diagnose: bool = False,
    max_batch_tokens: int = 0,
    max_batch_rows: int = 4,
    timeout_seconds: int = 1200,
    output: str = "torchtitan-benchmark.json",
):
    if not 120 <= timeout_seconds <= 2700:
        raise ValueError("timeout_seconds must be between 120 and 2700 (45 minutes)")
    result = benchmark.with_options(
        timeout=timeout_seconds, max_containers=1, retries=0
    ).remote(
        model,
        flavor,
        sequence_length,
        adapter_counts,
        steps,
        compile_layers,
        diagnose,
        max_batch_tokens,
        max_batch_rows,
        timeout_seconds,
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
