"""Publication plots for all completed screening controls, without selection by rank."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_results import rolling, save


ORDER = [
    "kv",
    "least-load",
    "bounded-kv",
    "adapter-home",
    "adapter-two",
    "adapter-spill",
    "balanced-home",
    "balanced-spill",
    "model-home",
    "model-spill",
    "lora",
    "work",
    "stable",
]
LABELS = dict(
    zip(
        ORDER,
        [
            "KV affinity",
            "Least load",
            "Bounded KV",
            "Static slots",
            "Two choices",
            "Slot overflow",
            "Even static",
            "Even overflow",
            "Job static",
            "Job overflow",
            "Legacy LoRA",
            "Work",
            "Domains",
        ],
        strict=True,
    )
)
COLORS = dict(zip(ORDER, plt.get_cmap("tab20")(np.arange(len(ORDER))), strict=True))


def label(arm):
    base = arm.removesuffix("-cohort")
    return LABELS.get(base, base) + (" + P" if arm.endswith("-cohort") else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    grouped = {}
    for job, arms in json.loads(args.metrics.read_text()).items():
        if not job.startswith("screen-"):
            continue
        target = grouped.setdefault(job.removesuffix("-supplement"), {})
        assert not (target.keys() & arms.keys())
        target.update(arms)
    for job, arms in grouped.items():
        assert len({r["workload_sha256"] for r in arms.values()}) == 1
        ordered = sorted(arms, key=lambda a: ORDER.index(a.removesuffix("-cohort")))
        colors = [COLORS[a.removesuffix("-cohort")] for a in ordered]
        labels = [label(a) for a in ordered]
        fig, axes = plt.subplots(
            2, 3, figsize=(12, max(6, len(ordered) * 0.57)), layout="constrained"
        )
        panels = [
            ("tokens_per_second", "Output tokens/s"),
            ("latency_mean_s", "Mean completion time (s)"),
            ("latency_p95_s", "p95 completion time (s)"),
            ("kv_hit_ratio", "KV token hit ratio"),
            ("request_lora_hit_ratio", "Hot-LoRA admission ratio"),
            ("gpu_loads", "GPU adapter loads"),
        ]
        for ax, (key, title) in zip(axes.flat, panels, strict=True):
            if key == "request_lora_hit_ratio":
                values = [
                    arms[a]["request_lora_hits"]
                    / (arms[a]["request_lora_hits"] + arms[a]["request_lora_misses"])
                    for a in ordered
                ]
            else:
                values = [arms[a][key] for a in ordered]
            ax.barh(labels, values, color=colors)
            ax.invert_yaxis()
            ax.set_xlabel(title)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle(job + ": one screening trial per policy")
        save(fig, args.output / (job + "-controls"))

        fig, axes = plt.subplots(
            1, 3, figsize=(12, max(3.5, len(ordered) * 0.3)), layout="constrained"
        )
        for ax, (key, title) in zip(
            axes,
            [
                ("native_busy_load_cv", "Request-count imbalance (CV)"),
                (
                    "mean_busy_active_adapter_versions_per_replica",
                    "Active versions / busy replica",
                ),
                (
                    "native_mean_running_requests_when_queued",
                    "Running requests while queued",
                ),
            ],
            strict=True,
        ):
            ax.barh(labels, [arms[a][key] for a in ordered], color=colors)
            ax.invert_yaxis()
            ax.set_xlabel(title)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
        save(fig, args.output / (job + "-mechanism"))

        # Readability: show the original policy and locality controls separately
        # from request-count/score controls; every completed policy is retained.
        for suffix, subset in [
            (
                "locality",
                {
                    "kv",
                    "adapter-home",
                    "adapter-two",
                    "adapter-spill",
                    "balanced-home",
                    "balanced-spill",
                    "model-home",
                    "model-spill",
                    "stable",
                },
            ),
            ("load", {"kv", "least-load", "bounded-kv", "lora", "work", "stable"}),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(10, 5.7), layout="constrained")
            for arm in ordered:
                base = arm.removesuffix("-cohort")
                if base not in subset:
                    continue
                data = json.loads(Path(arms[arm]["raw_path"]).read_text())
                times, series = rolling(data)
                for ax, key in zip(
                    axes.flat, ("cache", "latency", "throughput", "lora"), strict=True
                ):
                    ax.plot(
                        times,
                        series[key],
                        color=COLORS[base],
                        label=label(arm),
                        linestyle="--" if base == "kv" else "-",
                    )
            for ax, title in zip(
                axes.flat,
                (
                    "KV token hit ratio",
                    "Mean completion time (s)",
                    "Output tokens/s",
                    "Hot-LoRA admission ratio",
                ),
                strict=True,
            ):
                ax.set_ylabel(title)
                ax.set_xlabel("Time (s)")
                ax.grid(alpha=0.2)
            axes[0, 0].set_ylim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            handles, legends = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legends, loc="outside upper center", ncol=4)
            save(fig, args.output / (job + "-timeseries-" + suffix))


if __name__ == "__main__":
    main()
