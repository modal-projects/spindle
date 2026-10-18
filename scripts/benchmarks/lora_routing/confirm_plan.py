"""Freeze a selected policy and create counterbalanced held-out comparisons.

This writes a plan; it does not deploy or launch GPU work. Each comparison must
run its arms sequentially on one unchanged pool. Independent comparisons
may be assigned to separately monitored pools with --partition.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess
import time

from supervise import write


SOURCES = (
    "src/spindle/inference/lora_routing.py",
    "src/spindle/inference/routing_work.py",
    "src/spindle/inference/routing_domains.py",
    "src/spindle/inference/routing_simple.py",
    "src/spindle/inference/routing_capacity.py",
    "src/spindle/inference/sampling.py",
    "src/spindle/providers/modal/lora_routing.py",
    "scripts/benchmarks/lora_routing/cache_replay.py",
    "scripts/benchmarks/lora_routing/cohort_prefill.py",
    "scripts/benchmarks/lora_routing/h2_diagnostics.py",
    "src/spindle/inference/lora_sidecar.py",
    "src/spindle/inference/sglang_routing_observer.py",
    "scripts/benchmarks/lora_routing/cache_instrumentation.py",
    "scripts/benchmarks/lora_routing/cache_deployment.py",
    "scripts/benchmarks/lora_routing/controls_suite.py",
)


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in SOURCES}


def make_jobs(candidate, controls, trials=5, seed_start=14101):
    policies = ("kv", *controls, candidate)
    if len(set(policies)) != len(policies) or any(
        p.endswith("-cohort") for p in policies
    ):
        raise ValueError("Supply distinct base policies, without -cohort")
    # Rotate the order so each arm appears in each position before reuse.
    orders = [policies[i:] + policies[:i] for i in range(len(policies))]
    workloads = {
        "serving64": (
            "clients64.json",
            [
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
            ],
            1800,
        ),
        "long1024": (
            "clients16.json",
            [
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
                "3",
                "--warm-prefixes",
                "--cold-versions",
                "--large-group-clients",
                "8",
                "--hot-samples",
                "128",
            ],
            3600,
        ),
        "dialogue1024": (
            "clients16.json",
            [
                "--workload",
                "dapo",
                "--long-prefix",
                "--repeat-prefixes",
                "--stagger-seconds",
                "0.5",
                "--decode-tokens",
                "1024",
                "--groups",
                "4",
                "--samples",
                "8",
                "--rounds",
                "3",
                "--warm-prefixes",
                "--cold-versions",
                "--dialogue-turns",
                "3",
            ],
            3600,
        ),
    }
    jobs = []
    for trial in range(trials):
        for index, (name, (clients, arguments, minimum_remaining)) in enumerate(
            workloads.items()
        ):
            order = orders[(trial + index) % len(orders)]
            jobs.append(
                dict(
                    name=f"confirm-{name}-r{trial}",
                    clients=clients,
                    minimum_remaining_seconds=minimum_remaining * len(policies) / 3,
                    arguments=arguments
                    + ["--seed", str(seed_start + trial), "--vary-placement"],
                    arms=list(order),
                    trial=trial,
                    family=name,
                )
            )
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--trials", type=int, choices=(3, 5), default=5)
    parser.add_argument("--seed-start", type=int, default=14101)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", type=int, default=1)
    parser.add_argument("--http2", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists() or args.partition < 1:
        raise ValueError("Use a new output directory and a positive partition count")
    jobs = make_jobs(args.candidate, args.controls, args.trials, args.seed_start)
    for job in jobs:
        job["http2"] = args.http2
    args.output.mkdir(parents=True)
    frozen = dict(
        frozen_at=time.time(),
        candidate=args.candidate,
        controls=args.controls,
        trials=args.trials,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=source_hashes(),
        interpretation=f"{args.trials} held-out paired trials per family; report every trial. Screening data do not enter confirmatory estimates.",
        jobs=jobs,
    )
    write(args.output / "freeze.json", frozen)
    for partition in range(args.partition):
        start = len(jobs) * partition // args.partition
        end = len(jobs) * (partition + 1) // args.partition
        write(args.output / f"jobs-{partition}.json", jobs[start:end])


if __name__ == "__main__":
    main()
