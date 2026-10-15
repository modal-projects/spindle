"""Separate continuation reuse from adapter-version turnover in a replay."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmarks.lora_routing.cache_report import COLORS, LABELS, STYLES


def render(root, arms):
    rows, hashes = [], set()
    for arm in arms:
        data = json.loads((root / f"{arm}.json").read_text())
        hashes.add(data["workload_sha256"])
        for dimension in ["turn", "dialogue_turn"]:
            values = sorted({r[dimension] for r in data["requests"]})
            for value in values:
                requests = [r for r in data["requests"] if r[dimension] == value]
                rows.append(
                    dict(
                        policy=arm,
                        dimension=dimension,
                        index=value,
                        requests=len(requests),
                        versions=sorted({r["version"] for r in requests}),
                        kv_hit_ratio=sum(r["cached_tokens"] for r in requests)
                        / sum(r["prompt_tokens"] for r in requests),
                        latency_mean_s=float(np.mean([r["seconds"] for r in requests])),
                        latency_p95_s=float(
                            np.quantile([r["seconds"] for r in requests], 0.95)
                        ),
                    )
                )
        del data, requests
    assert len(hashes) == 1, "Different workload configurations"
    (root / "continuations.json").write_text(json.dumps(rows, indent=2))
    with (root / "continuations.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for column, dimension in enumerate(["dialogue_turn", "turn"]):
        for arm in arms:
            selected = [
                r for r in rows if r["policy"] == arm and r["dimension"] == dimension
            ]
            xs = [r["index"] for r in selected]
            for ax, metric in [
                (axes[0, column], "kv_hit_ratio"),
                (axes[1, column], "latency_mean_s"),
            ]:
                ax.plot(
                    xs,
                    [r[metric] for r in selected],
                    marker="o",
                    color=COLORS[arm],
                    linestyle=STYLES[arm],
                    label=LABELS[arm],
                )
                ax.set_xticks(
                    xs,
                    [
                        str(r["index"])
                        + (
                            "\nv" + ",".join(map(str, r["versions"]))
                            if dimension == "turn"
                            else ""
                        )
                        for r in selected
                    ],
                )
                ax.set_xlabel(
                    "Conversation turn (0 = initial prompt)"
                    if dimension == "dialogue_turn"
                    else "Independent rollout round and served versions"
                )
                ax.grid(alpha=0.25)
        axes[0, column].set(ylabel="KV token hit ratio", ylim=(0, 1))
        axes[1, column].set(ylabel="Mean request completion time (seconds)")
    axes[0, 0].legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"continuations.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--arms", nargs="+", default=["kv", "model-spill", "dlora-no-merge"]
    )
    args = parser.parse_args()
    render(args.root, args.arms)
