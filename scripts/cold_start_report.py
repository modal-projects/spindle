"""Summarize where a cold start spent its time from Modal app logs.

Usage:

    uv run modal app logs spindle-trainer-<config> > trainer.log
    uv run modal app logs spindle-lora-<definition id> > sampler.log
    uv run scripts/cold_start_report.py trainer.log sampler.log

Reads the ``spindle_startup_mark`` lines that every Spindle container prints,
plus SGLang's own startup timings, and prints one phase table per container.
Pass ``--json`` to get the same data as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

MARK = re.compile(r'\{"event": "spindle_startup_mark".*?\}(?=\s*$)')
STEP = re.compile(r'\{"event": "spindle_step_timing".*?\}(?=\s*$)')
TASK = re.compile(r"\b(ta-[0-9A-HJKMNP-TV-Z]{26})\b")
SGLANG = [
    (
        "sglang_init_distributed",
        re.compile(r"Init torch distributed ends\. elapsed=([\d.]+)"),
    ),
    ("sglang_load_weight", re.compile(r"Load weight end\. elapsed=([\d.]+)")),
    (
        "sglang_cuda_graph_prefill",
        re.compile(r"Capture target prefill CUDA graph end\. elapsed=([\d.]+)"),
    ),
    (
        "sglang_cuda_graph_decode",
        re.compile(r"Capture target decode CUDA graph end\. elapsed=([\d.]+)"),
    ),
    ("sglang_cuda_graph", re.compile(r"Capture cuda graph end\. .*?elapsed=([\d.]+)")),
]
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def task_created_at(task_id: str) -> float:
    """Modal task ids are ULIDs; the first ten characters encode creation time."""
    value = 0
    for char in task_id[3:13]:
        value = value * 32 + CROCKFORD.index(char)
    return value / 1000


def parse(path: str) -> dict[str, dict]:
    containers: dict[str, dict] = defaultdict(
        lambda: {"marks": [], "steps": [], "sglang": {}, "created_at": None}
    )
    with open(path, errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            task = TASK.search(line)
            key = task.group(1) if task else "unknown"
            container = containers[key]
            if task and container["created_at"] is None:
                container["created_at"] = task_created_at(key)
            if match := MARK.search(line):
                container["marks"].append(json.loads(match.group(0)))
                continue
            if match := STEP.search(line):
                container["steps"].append(json.loads(match.group(0)))
                continue
            for name, pattern in SGLANG:
                if match := pattern.search(line):
                    container["sglang"][name] = float(match.group(1))
    return {
        key: value
        for key, value in containers.items()
        if value["marks"] or value["sglang"] or value["steps"]
    }


def summarize(container: dict) -> dict:
    marks = container["marks"]
    by_component: dict[str, list[dict]] = defaultdict(list)
    for mark in marks:
        by_component[mark["component"]].append(mark)
    phases = []
    for component, items in by_component.items():
        items.sort(key=lambda item: item["ts"])
        for item in items:
            phases.append(
                {
                    "component": component,
                    "mark": item["mark"],
                    "phase_s": item["phase_s"],
                    "since_start_s": item["since_start_s"],
                    "at": datetime.fromtimestamp(item["ts"], timezone.utc).strftime(
                        "%H:%M:%S"
                    ),
                    "extra": {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "event",
                            "component",
                            "mark",
                            "ts",
                            "since_start_s",
                            "phase_s",
                        }
                    },
                }
            )
    first_mark = min(
        (mark["ts"] - mark["since_start_s"] for mark in marks), default=None
    )
    created = container["created_at"]
    steps = defaultdict(dict)
    for step in container["steps"]:
        steps[step["step"]][step["phase"]] = step["seconds"]
    return {
        "created_at": created,
        "boot_to_process_s": None
        if created is None or first_mark is None
        else round(first_mark - created, 1),
        "phases": sorted(phases, key=lambda phase: phase["at"]),
        "sglang": container["sglang"],
        "steps": {str(key): value for key, value in sorted(steps.items())},
    }


def render(task_id: str, summary: dict) -> str:
    lines = [f"## {task_id}"]
    if summary["boot_to_process_s"] is not None:
        lines.append(
            f"container created -> first process start: {summary['boot_to_process_s']} s"
        )
    lines.append("")
    lines.append(
        f"{'component':16} {'mark':34} {'phase s':>9} {'since start s':>14}  at (UTC)"
    )
    for phase in summary["phases"]:
        extra = ", ".join(f"{k}={v}" for k, v in phase["extra"].items())
        lines.append(
            f"{phase['component']:16} {phase['mark']:34} {phase['phase_s']:9.1f} "
            f"{phase['since_start_s']:14.1f}  {phase['at']}  {extra}"
        )
    if summary["sglang"]:
        lines.append("")
        lines.append("SGLang startup timings (s):")
        for name, seconds in summary["sglang"].items():
            lines.append(f"  {name:30} {seconds:8.1f}")
    if summary["steps"]:
        lines.append("")
        lines.append("Training step phases (s):")
        for step, phases in summary["steps"].items():
            parts = ", ".join(
                f"{name}={seconds:.1f}" for name, seconds in phases.items()
            )
            lines.append(f"  step {step}: {parts}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logs", nargs="+", help="Modal app log files")
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of text"
    )
    args = parser.parse_args()
    report = {}
    for path in args.logs:
        for task_id, container in parse(path).items():
            report[task_id] = summarize(container)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        return
    if not report:
        print("no spindle_startup_mark lines found")
        return
    print("\n\n".join(render(task_id, summary) for task_id, summary in report.items()))


if __name__ == "__main__":
    main()
