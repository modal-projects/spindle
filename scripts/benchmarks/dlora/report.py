"""Validate paired fixed-work trials and plot cache, mode, and batch measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmarks.lora_routing.cache_report import (
    COLORS,
    LABELS,
    STYLES,
    summarize,
)


def counters(states):
    keys = {
        k
        for s in states
        for k, value in s["gpu_lora"].items()
        if isinstance(value, (int, float))
    }
    return {k: sum(s["gpu_lora"].get(k, 0) for s in states) for k in keys}


def metrics(data):
    result = summarize(data)
    before, after = counters(data["baseline"]), counters(data["final"])
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    steps = sum(
        delta.get(k, 0) for k in ["merged_decode_steps", "unmerged_decode_steps"]
    )
    mode_totals = {}
    for key in ["count", "seconds"]:
        mode_totals[key] = sum(
            s["mode"]["transition_totals"].get(key, 0) for s in data["final"]
        ) - sum(s["mode"]["transition_totals"].get(key, 0) for s in data["baseline"])
    result.update(
        merged_requests=sum(
            any(
                a["attrs"].get("ok") and a["attrs"].get("lora_merged")
                for a in r["attempts"]
            )
            for r in data["requests"]
        ),
        merged_decode_step_fraction=delta.get("merged_decode_steps", 0) / max(1, steps),
        observed_decode_steps=steps,
        mean_executed_decode_batch=delta.get("decode_batch_sequences", 0)
        / max(1, steps),
        mean_distinct_adapters_per_decode_batch=delta.get(
            "decode_batch_distinct_adapters", 0
        )
        / max(1, steps),
        sequence_weighted_dominant_adapter_fraction=delta.get(
            "decode_batch_dominant_sequences", 0
        )
        / max(1, delta.get("decode_batch_sequences", 0)),
        mode_transitions=mode_totals["count"],
        mode_transition_replica_seconds=mode_totals["seconds"],
        responses_503=sum(
            a["attrs"].get("http_status") == 503
            for r in data["requests"]
            for a in r["attempts"]
        ),
        transport_errors=sum(
            bool(a["attrs"].get("error_type"))
            for r in data["requests"]
            for a in r["attempts"]
        ),
    )
    return result


def render(root, arms):
    paths = [root / (arm + ".json") for arm in arms]
    data = {p.stem: json.loads(p.read_text()) for p in paths}
    assert len({d["workload_sha256"] for d in data.values()}) == 1
    summary = {arm: metrics(d) for arm, d in data.items()}
    (root / "dlora-summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "dlora-summary.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["policy", *next(iter(summary.values()))]
        )
        writer.writeheader()
        writer.writerows({"policy": k, **v} for k, v in summary.items())
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    for arm, d in data.items():
        kwargs = dict(label=LABELS[arm], color=COLORS[arm], linestyle=STYLES[arm])
        bins = np.arange(0, d["seconds"] + 20, 20)
        rows = d["requests"]
        for column, metric in [(0, "kv"), (1, "latency")]:
            ys = []
            for lo, hi in zip(bins, bins[1:]):
                window = [r for r in rows if lo <= r["end"] < hi]
                ys.append(
                    (
                        sum(r["cached_tokens"] for r in window)
                        / sum(r["prompt_tokens"] for r in window)
                        if metric == "kv"
                        else np.mean([r["seconds"] for r in window])
                    )
                    if window
                    else np.nan
                )
            axes[0, column].plot((bins[:-1] + bins[1:]) / 2, ys, **kwargs)
        snapshots = (
            [(0, d["baseline"])]
            + [
                (o["elapsed"], o["replicas"])
                for o in d["observations"]
                if "replicas" in o
            ]
            + [(d["seconds"], d["final"])]
        )
        xs, hits, batches, merged, loads = [], [], [], [], []
        start = counters(d["baseline"])
        for (t0, s0), (t1, s1) in zip(snapshots, snapshots[1:]):
            a, b = counters(s0), counters(s1)
            change = {k: b.get(k, 0) - a.get(k, 0) for k in b}
            n = change.get("request_lora_hits", 0) + change.get(
                "request_lora_misses", 0
            )
            steps = change.get("merged_decode_steps", 0) + change.get(
                "unmerged_decode_steps", 0
            )
            xs.append((t0 + t1) / 2)
            hits.append(change.get("request_lora_hits", 0) / n if n else np.nan)
            batches.append(
                change.get("decode_batch_sequences", 0) / steps if steps else np.nan
            )
            merged.append(
                change.get("merged_decode_steps", 0) / steps if steps else np.nan
            )
            loads.append(b.get("gpu_loads", 0) - start.get("gpu_loads", 0))
        for ax, values in [
            (axes[1, 0], hits),
            (axes[1, 1], batches),
            (axes[2, 0], merged),
            (axes[2, 1], loads),
        ]:
            ax.plot(xs, values, **kwargs)
    labels = [
        "KV token hit ratio",
        "Mean request completion time (s)",
        "GPU LoRA-buffer hit ratio",
        "Mean executed decode batch",
        "Merged decode-step fraction",
        "Cumulative GPU adapter loads",
    ]
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel="Time (seconds)", ylabel=label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / ("dlora-timeseries." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument(
        "--arms", nargs="+", default=["kv", "model-spill", "dlora-no-merge", "dlora"]
    )
    args = parser.parse_args()
    for root in args.roots:
        render(root, args.arms)
