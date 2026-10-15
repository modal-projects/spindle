"""Compare completed workload summaries without averaging distinct workloads."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from scripts.benchmarks.lora_routing.cache_report import COLORS, LABELS


def render(manifest, output):
    trials = json.loads(manifest.read_text())
    rows = []
    arms = ["kv", "model-spill", "dlora-no-merge", "dlora"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for index, trial in enumerate(trials):
        root = Path(trial["directory"])
        completed = json.loads((root / "dlora-summary.json").read_text())
        summaries = {arm: completed[arm] for arm in trial["arms"]}
        baseline = summaries["kv"]
        for arm, summary in summaries.items():
            assert summary["requests"] == baseline["requests"]
            assert summary["output_tokens"] == baseline["output_tokens"]
            assert summary["extra_engine_admissions"] == 0, (
                "Put trials with duplicate native work in a separate incident report"
            )
            throughput_ratio = (
                summary["generation_tokens_per_second"]
                / baseline["generation_tokens_per_second"]
            )
            latency_ratio = summary["latency_p95_s"] / baseline["latency_p95_s"]
            rows.append(
                dict(
                    trial=trial["label"],
                    directory=str(root),
                    policy=arm,
                    throughput_relative_to_kv=throughput_ratio,
                    p95_relative_to_kv=latency_ratio,
                    **summary,
                )
            )
            position = index + (arms.index(arm) - 1.5) * 0.19
            for ax, value in zip(axes, [throughput_ratio, latency_ratio]):
                ax.bar(position, value, width=0.18, color=COLORS[arm])
                ax.annotate(
                    f"{value:.2f}",
                    (position, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    for ax, label in zip(
        axes,
        [
            "Output throughput / KV baseline (higher is better)",
            "p95 request latency / KV baseline (lower is better)",
        ],
    ):
        ax.axhline(1, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(np.arange(len(trials)), [trial["label"] for trial in trials])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    axes[0].legend(
        handles=[Patch(color=COLORS[arm], label=LABELS[arm]) for arm in arms],
        fontsize=8,
        ncols=2,
    )
    fig.suptitle("Fixed-work trials: individual measurements, not confidence intervals")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    columns = [
        "trial",
        "policy",
        "directory",
        "requests",
        "output_tokens",
        "generation_makespan_seconds",
        "generation_tokens_per_second",
        "latency_mean_s",
        "latency_p95_s",
        "throughput_relative_to_kv",
        "p95_relative_to_kv",
        "kv_hit_ratio",
        "lora_hit_ratio",
        "gpu_loads",
        "gpu_evictions",
        "mean_executed_decode_batch",
        "merged_requests",
        "merged_decode_step_fraction",
        "responses_503",
        "transport_errors",
        "extra_engine_admissions",
    ]
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix("." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.manifest, args.output)
