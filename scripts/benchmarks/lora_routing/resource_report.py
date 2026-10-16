"""Deduplicate explicitly supplied GPU lifetime manifests, including failures."""

import argparse
import json
from pathlib import Path
import time

from supervise import write


def summarize(paths, as_of):
    tasks = {}
    for path in paths:
        for row in json.loads(path.read_text()):
            if not row["gpu_count"]:
                continue
            old = tasks.get(row["task_id"])
            if old is not None:
                for key in ("gpu_count", "gpu_type"):
                    if old[key] != row[key]:
                        raise ValueError(f"Conflicting {key} for {row['task_id']}")
                if (
                    old["started_at"]
                    and row["started_at"]
                    and old["started_at"] != row["started_at"]
                ):
                    raise ValueError(f"Conflicting start for {row['task_id']}")
                old["started_at"] = old["started_at"] or row["started_at"]
                old["finished_at"] = max(old["finished_at"], row["finished_at"])
            else:
                tasks[row["task_id"]] = dict(row)
    events, hours = [], {}
    active = 0
    unstarted = []
    for row in tasks.values():
        start = row["started_at"]
        if not start:
            unstarted.append(row["task_id"])
            continue
        end = min(row["finished_at"] or as_of, as_of)
        if start > end:
            raise ValueError("Lifetime starts after report cutoff or recorded end")
        count = row["gpu_count"]
        hours[row["gpu_type"]] = (
            hours.get(row["gpu_type"], 0) + count * (end - start) / 3600
        )
        events.extend([(start, count), (end, -count)])
        if not row["finished_at"] or row["finished_at"] > as_of:
            active += count
    concurrent = peak = 0
    for _, change in sorted(events):
        concurrent += change
        peak = max(peak, concurrent)
    return dict(
        as_of=as_of,
        gpu_hours=sum(hours.values()),
        gpu_hours_by_type=hours,
        peak_gpus=peak,
        observed_active_gpus=active,
        task_count=len(tasks),
        tasks_without_recorded_start=unstarted,
        sources=[str(p) for p in paths],
        tasks=list(tasks.values()),
        interpretation="Deduplicated started task lifetimes in the explicitly listed manifests, including setup and failed attempts. Running tasks end at the report cutoff. Queued tasks without a recorded start are listed separately and contribute no measured runtime. This is not a billing or GPU-utilization measurement; it does not establish coverage of unlisted experiments.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--as-of", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.manifests, args.as_of or time.time())
    write(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"tasks", "sources"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
