"""Run the two real workloads sequentially; each uses the established client supervisor."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.benchmarks.lora_routing.recovery import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["dapo", "codeforces"],
        default=["dapo", "codeforces"],
    )
    args = parser.parse_args()
    status = dict(started=time.time(), complete=False)
    try:
        for workload in args.workloads:
            root = args.root / workload
            status.update(workload=workload, phase="deploying")
            atomic_json(args.root / "study-status.json", status)
            for policy in json.loads((root / "schedule.json").read_text())["pairs"][0]:
                with (root / f"deploy-{policy}.log").open("a") as output:
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.benchmarks.rl_rust.deploy",
                            "--root",
                            str(root),
                            "--policy",
                            policy,
                        ],
                        check=True,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                    )
            status["phase"] = "running"
            atomic_json(args.root / "study-status.json", status)
            with (root / "supervisor.log").open("a") as output:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.benchmarks.rl_rust.run",
                        "--root",
                        str(root),
                        "--external-root",
                        str(args.external_root),
                    ],
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=13 * 3600,
                )
            if not json.loads((root / "status.json").read_text())["complete"]:
                raise RuntimeError(f"{workload} did not complete")
        status["complete"] = True
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["finished"] = time.time()
        atomic_json(args.root / "study-status.json", status)


if __name__ == "__main__":
    main()
