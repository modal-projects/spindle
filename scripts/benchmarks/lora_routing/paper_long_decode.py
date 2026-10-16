"""Run one longer-output control only after the main suite exits successfully."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--duration-seconds", type=int, default=300)
    args = parser.parse_args()
    process = Path(f"/proc/{args.after_pid}/cmdline")
    while process.exists():
        command = process.read_bytes().replace(b"\0", b" ").decode()
        if "paper_suite.py" not in command or str(args.root) not in command:
            break
        if (args.root / "STOP").exists():
            raise RuntimeError("Pool shutdown requested; longer-output control skipped")
        time.sleep(10)
    previous = json.loads((args.root / "suite-http2-manifest.json").read_text())
    if len(previous) != 9 or any(r["status"] != "complete" for r in previous):
        raise RuntimeError(
            "Main suite did not finish; inspect before starting another job"
        )
    deadline = json.loads((args.root / "monitor.json").read_text())["deadline"]
    # Allow five minutes per arm for reset, warmup, and draining work beyond
    # the arrival window. The independent monitor remains the hard deadline.
    if deadline - time.time() < 2 * args.duration_seconds + 600:
        raise RuntimeError(
            "Insufficient bounded allocation for the longer-output control"
        )
    output = args.root / "long-decode1024-r0"
    if output.exists():
        raise RuntimeError("Refusing to overwrite an existing longer-output result")
    command = [
        sys.executable,
        "scripts/benchmarks/lora_routing/cache_replay.py",
        "--pools",
        str(args.root / "pools.json"),
        "--clients",
        str(args.clients),
        "--work-rates",
        str(args.root / "calibration/work-rates.json"),
        "--output",
        str(output),
        "--arms",
        "kv-cohort",
        "stable-cohort",
        "--http2",
        "--request-timeout",
        "600",
        "--sequential",
        "--keep-pools",
        "--workload",
        "dapo",
        "--long-prefix",
        "--repeat-prefixes",
        "--stagger-seconds",
        "0.5",
        "--decode-tokens",
        "1024",
        "--groups",
        "8",
        "--samples",
        "8",
        "--rounds",
        "100",
        "--duration-seconds",
        str(args.duration_seconds),
        "--warm-prefixes",
        "--cold-versions",
        "--large-group-clients",
        "8",
        "--hot-samples",
        "128",
    ]
    target = args.root / "long-decode1024-manifest.json"
    record = dict(command=command, start=time.time(), status="running")
    target.write_text(json.dumps(record, indent=2) + "\n")
    with (args.root / "long-decode1024-r0.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    record.update(
        end=time.time(),
        returncode=result.returncode,
        status="complete" if result.returncode == 0 else "failed",
    )
    target.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("Longer-output control failed; inspect its preserved log")


if __name__ == "__main__":
    main()
