"""Run the prespecified paper comparisons sequentially on an existing pool.

This launcher allocates no GPUs. An independently running cache_monitor owns
the pool lifetime. Every completed arm is retained before starting the next.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    parser.add_argument("--phase", choices=("initial", "http2"), default="initial")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = [
        sys.executable,
        "scripts/benchmarks/lora_routing/cache_replay.py",
        "--pools",
        str(args.root / "pools.json"),
        "--work-rates",
        str(args.root / "calibration/work-rates.json"),
        "--request-timeout",
        "600",
        "--sequential",
        "--keep-pools",
    ]
    rl = [
        "--workload",
        "dapo",
        "--long-prefix",
        "--repeat-prefixes",
        "--stagger-seconds",
        "0.5",
        "--decode-tokens",
        "256",
    ]
    shapes = {
        "warm": rl
        + [
            "--groups",
            "8",
            "--samples",
            "8",
            "--rounds",
            "4",
            "--warm-prefixes",
            "--fixed-version",
        ],
        "turnover": rl
        + [
            "--groups",
            "8",
            "--samples",
            "8",
            "--rounds",
            "4",
            "--warm-prefixes",
            "--cold-versions",
            "--large-group-clients",
            "8",
            "--hot-samples",
            "128",
        ],
        "dialogue": rl
        + [
            "--groups",
            "4",
            "--samples",
            "8",
            "--rounds",
            "3",
            "--dialogue-turns",
            "3",
            "--warm-prefixes",
            "--cold-versions",
        ],
    }
    jobs = [
        (
            "turnover-r0",
            "turnover",
            ["kv", "kv-cohort", "stable-cohort", "work-cohort"],
        ),
        ("warm-r0", "warm", ["kv", "kv-cohort", "stable-cohort"]),
        ("dialogue-r0", "dialogue", ["kv", "kv-cohort", "stable-cohort"]),
        ("turnover-r1", "turnover", ["stable-cohort", "kv-cohort"]),
        ("warm-r1", "warm", ["stable-cohort", "kv-cohort"]),
        ("dialogue-r1", "dialogue", ["stable-cohort", "kv-cohort"]),
    ]
    if args.phase == "http2":
        common.append("--http2")
        # Three warmed rounds permit independent reversed-order repeats within
        # the bounded allocation; turnover retains two rounds per version.
        shapes["warm"][shapes["warm"].index("--rounds") + 1] = "3"
        shapes["general"] = [
            "--streams",
            "8",
            "--conversations",
            "4",
            "--turns",
            "4",
            "--prompt-tokens",
            "8192",
            "--shared-tokens",
            "7168",
            "--decode-tokens",
            "512",
        ]
        shapes["long"] = list(shapes["turnover"])
        shapes["long"][shapes["long"].index("--decode-tokens") + 1] = "512"
        shapes["long"][shapes["long"].index("--rounds") + 1] = "100"
        shapes["long"] += ["--duration-seconds", "300"]
        jobs = [
            ("warm-h2-r0", "warm", ["kv-cohort", "stable-cohort"]),
            ("dialogue-h2-r0", "dialogue", ["kv-cohort", "stable-cohort"]),
            ("serving64-final-r0", "general", ["default", "kv", "stable"]),
            ("turnover-h2-r0", "turnover", ["kv-cohort", "stable-cohort"]),
            ("warm-h2-r1", "warm", ["stable-cohort", "kv-cohort"]),
            ("serving64-final-r1", "general", ["stable", "kv"]),
            ("dialogue-h2-r1", "dialogue", ["stable-cohort", "kv-cohort"]),
            ("turnover-h2-r1", "turnover", ["stable-cohort", "kv-cohort"]),
            ("long-decode-r0", "long", ["kv-cohort", "stable-cohort"]),
        ]
        command = [
            sys.executable,
            "scripts/benchmarks/lora_routing/cohort_check.py",
            "--pools",
            str(args.root / "pools.json"),
            "--clients",
            str(args.clients),
            "--http2",
            "--output",
            str(args.root / "cohort-http2-correctness.json"),
        ]
        if not args.resume:
            with (args.root / "http2-correctness.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
    target = args.root / (
        "suite-http2-manifest.json" if args.phase == "http2" else "suite-manifest.json"
    )
    manifest = json.loads(target.read_text()) if args.resume else []
    if any(r["status"] != "complete" for r in manifest):
        raise RuntimeError("Preserve and resolve interrupted records before resuming")
    completed = {r["name"] for r in manifest}
    for name, shape, arms in jobs:
        if name in completed:
            continue
        deadline = json.loads((args.root / "monitor.json").read_text())["deadline"]
        if deadline - time.time() < (
            1200 if shape == "long" or len(arms) == 3 else 900
        ):
            raise RuntimeError("Compute deadline approaching; no new job started")
        command = (
            common
            + [
                "--clients",
                str(
                    args.root / "clients64.json" if shape == "general" else args.clients
                ),
            ]
            + shapes[shape]
            + ["--output", str(args.root / name), "--arms", *arms]
        )
        row = dict(name=name, command=command, start=time.time(), status="running")
        manifest.append(row)
        target.write_text(json.dumps(manifest, indent=2))
        with (args.root / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False,
            )
        row.update(
            end=time.time(),
            returncode=result.returncode,
            status="complete" if result.returncode == 0 else "failed",
        )
        target.write_text(json.dumps(manifest, indent=2))
        if result.returncode:
            raise RuntimeError(
                f"{name} failed; inspect its preserved log before recovery"
            )


if __name__ == "__main__":
    main()
