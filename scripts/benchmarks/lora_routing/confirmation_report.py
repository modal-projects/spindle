"""Summarize frozen paired trials without treating requests as replicates."""

import argparse
import json
import math
from pathlib import Path
import statistics

from controls_plots import label
from paper_results import analyzed
from supervise import write


METRICS = (
    "tokens_per_second",
    "latency_mean_s",
    "latency_p95_s",
    "rollout_mean_s",
    "gpu_loads",
    "gpu_evictions",
    "load_host_seconds",
    "kv_hit_ratio",
)


def summarize_pairs(frozen, roots):
    results, missing, ratios = {}, [], {}
    for job in frozen["jobs"]:
        folders = [
            root / job["name"] for root in roots if (root / job["name"]).exists()
        ]
        if len(folders) > 1:
            raise ValueError(f"Duplicate comparison: {job['name']}")
        if not folders:
            missing.append(job["name"])
            continue
        arms = {}
        for arm in job["arms"]:
            path = folders[0] / f"{arm}.json"
            if not path.exists():
                missing.append(f"{job['name']}/{arm}")
                continue
            raw = json.loads(path.read_text())
            if raw.get("error"):
                raise ValueError(f"Failed held-out arm: {path}: {raw['error']}")
            # The replay captures the same relative source paths as the freeze.
            recorded = raw["config"]["source_sha256"]
            for name, digest in recorded.items():
                if name in frozen["source_sha256"]:
                    if frozen["source_sha256"][name] != digest:
                        raise ValueError(f"Source changed in {path}: {name}")
            arms[arm] = analyzed(raw)
            arms[arm]["workload_sha256"] = raw["workload_sha256"]
        if len(arms) != len(job["arms"]):
            continue
        if len({a["workload_sha256"] for a in arms.values()}) != 1:
            raise ValueError(f"Unmatched workload: {job['name']}")
        results[job["name"]] = arms
        candidate = arms[frozen["candidate"]]
        for baseline in ("kv", *frozen["controls"]):
            for metric in METRICS:
                a, b = candidate[metric], arms[baseline][metric]
                if a is None or b in (None, 0):
                    continue
                key = f"{job['family']}/{baseline}/{metric}"
                ratios.setdefault(key, []).append(
                    dict(trial=job["trial"], candidate=a, baseline=b, ratio=a / b)
                )
    aggregates = {}
    for key, rows in ratios.items():
        values = [r["ratio"] for r in rows]
        estimate = dict(
            trials=rows,
            n=len(values),
            median_ratio=statistics.median(values),
            minimum_ratio=min(values),
            maximum_ratio=max(values),
        )
        if all(v > 0 for v in values):
            logs = [math.log(v) for v in values]
            mean = statistics.mean(logs)
            estimate["geometric_mean_ratio"] = math.exp(mean)
            if len(values) == frozen["trials"]:
                # A two-sided Student t interval. The unit is a
                # complete paired workload trial, never an individual request.
                critical = {3: 4.302652729696142, 5: 2.7764451051977987}[len(values)]
                radius = critical * statistics.stdev(logs) / math.sqrt(len(values))
                estimate["geometric_mean_ratio_95pct_interval"] = [
                    math.exp(mean - radius),
                    math.exp(mean + radius),
                ]
        aggregates[key] = estimate
    return dict(
        candidate=frozen["candidate"],
        controls=frozen["controls"],
        complete=not missing,
        missing=missing,
        paired=aggregates,
        runs=results,
        uncertainty=f"{frozen['trials']} paired workload seeds per family; t interval on log ratios assumes independent trial-level log ratios. All trials and ranges are retained. Requests within a trial are not independent replicates; exploratory screens are excluded.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_pairs(json.loads(args.freeze.read_text()), args.roots)
    write(args.output, result)
    print(f"Candidate: {label(result['candidate'])}; complete: {result['complete']}")
    for key, row in result["paired"].items():
        print(
            f"{key}: n={row['n']} ratios={[round(r['ratio'], 4) for r in row['trials']]}"
        )
    if result["missing"]:
        print("Missing: " + ", ".join(result["missing"]))


if __name__ == "__main__":
    main()
