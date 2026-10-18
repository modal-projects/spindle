"""Run recorded control comparisons without allocating any additional GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from supervise import write
from confirm_plan import source_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--manifest", default="controls-manifest.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    target = args.root / args.manifest
    records = json.loads(target.read_text()) if args.resume else []
    if target.exists() and not args.resume:
        raise RuntimeError("Manifest exists; preserve it and use --resume")
    if any(r["status"] != "complete" for r in records):
        raise RuntimeError("Resolve the recorded interruption before resuming")
    completed = {r["name"] for r in records}
    jobs = json.loads(args.jobs.read_text())
    for job in jobs:
        if job["name"] in completed:
            continue
        if args.freeze:
            frozen = json.loads(args.freeze.read_text())
            if source_hashes() != frozen["source_sha256"]:
                raise RuntimeError(
                    "Source changed after policy selection; preserve the frozen experiment"
                )
            if job not in frozen["jobs"]:
                raise RuntimeError("Comparison differs from the frozen plan")
        monitor = json.loads((args.root / "monitor.json").read_text())
        if time.time() - monitor["updated"] > 60:
            raise RuntimeError("Allocation monitor is not current")
        if monitor["deadline"] - time.time() < job["minimum_remaining_seconds"]:
            raise RuntimeError("No new comparison can fit the allocation deadline")
        command = [
            sys.executable,
            "scripts/benchmarks/lora_routing/cache_replay.py",
            "--pools",
            str(args.root / "pools.json"),
            "--clients",
            str(args.root / job["clients"]),
            "--work-rates",
            str(args.root / "calibration/work-rates.json"),
            "--output",
            str(args.root / job["name"]),
            "--request-timeout",
            "600",
            *(["--http2"] if job.get("http2", True) else []),
            "--sequential",
            "--keep-pools",
            *job["arguments"],
            "--arms",
            *job["arms"],
        ]
        record = dict(
            name=job["name"], command=command, start=time.time(), status="running"
        )
        records.append(record)
        write(target, records)
        with (args.root / (job["name"] + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "MODAL_ENVIRONMENT": "lilo-deploy"},
            )
        record.update(
            end=time.time(),
            returncode=result.returncode,
            status="complete" if result.returncode == 0 else "failed",
        )
        write(target, records)
        if result.returncode:
            raise RuntimeError(f"{job['name']} failed; inspect saved evidence")


if __name__ == "__main__":
    main()
