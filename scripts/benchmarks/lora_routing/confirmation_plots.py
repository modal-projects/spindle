"""Plot every frozen trial and a predeclared trial-zero time series."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cache_report import totals
from confirmation_report import summarize_pairs
from controls_plots import label
from paper_results import export_plot_values, rolling, save
from supervise import write


COLORS = {
    "kv": "#be4141",
    "balanced-home": "#5b8b49",
    "model-spill": "#c48626",
    "stable": "#2d61a5",
}
FAMILIES = {
    "serving64": "64 adapter versions",
    "long1024": "Independent long rollouts",
    "dialogue1024": "Three-turn rollouts",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.freeze.read_text())
    result = summarize_pairs(frozen, args.roots)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "paired-results.json", result)
    if not result["complete"]:
        raise ValueError(
            "Frozen study incomplete; saved partial analysis, no final plots"
        )
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    policies = ["kv", *frozen["controls"], frozen["candidate"]]
    rows, raw = [], {}
    for job in frozen["jobs"]:
        for policy, metrics in result["runs"][job["name"]].items():
            rows.append(
                dict(
                    family=job["family"],
                    trial=job["trial"],
                    policy=policy,
                    **{
                        k: v
                        for k, v in metrics.items()
                        if not isinstance(v, (dict, list))
                    },
                )
            )
            raw.setdefault(job["name"], {})[policy] = json.loads(
                Path(metrics["raw_path"]).read_text()
            )
    with (args.output / "all-trials.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    export_plot_values(raw, args.output, max_observation_gap=15)

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.4), layout="constrained")
    panels = [
        ("tokens_per_second", "Output tokens/s"),
        ("latency_mean_s", "Mean completion time (s)"),
        ("latency_p95_s", "p95 completion time (s)"),
    ]
    for column, family in enumerate(FAMILIES):
        for row, (metric, ylabel) in enumerate(panels):
            ax = axes[row, column]
            for index, policy in enumerate(policies):
                selected = [
                    r for r in rows if r["family"] == family and r["policy"] == policy
                ]
                values = [r[metric] for r in selected]
                offsets = np.linspace(-0.14, 0.14, len(values))
                ax.scatter(index + offsets, values, color=COLORS[policy], s=24)
                ax.plot(
                    [index - 0.2, index + 0.2],
                    [np.mean(values)] * 2,
                    color=COLORS[policy],
                    linewidth=2,
                )
            ax.set_xticks(
                range(len(policies)),
                [label(p) for p in policies],
                rotation=25,
                ha="right",
            )
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(FAMILIES[family])
    save(fig, args.output / "all-trials")

    mechanism_values = {}
    for family, title in FAMILIES.items():
        arms = raw[f"confirm-{family}-r0"]
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), layout="constrained")
        for policy in policies:
            data = arms[policy]
            x, curves = rolling(data, max_observation_gap=15)
            observation_times = [
                0,
                *[o["elapsed"] for o in data["observations"] if "replicas" in o],
                data["seconds"],
            ]
            gaps = [
                (a, b)
                for a, b in zip(observation_times, observation_times[1:])
                if b - a > 15
            ]
            style = dict(
                color=COLORS[policy],
                label=label(policy),
                linestyle="--" if policy == "kv" else "-",
            )
            for ax, metric in zip(
                axes.flat, ("cache", "lora", "latency", "throughput")
            ):
                ax.plot(x, curves[metric], **style)
            baseline = totals(data["baseline"])["gpu_loads"]
            polls = [o for o in data["observations"] if "replicas" in o]
            axes[1, 1].plot(
                [0, *[o["elapsed"] for o in polls], data["seconds"]],
                [
                    0,
                    *[totals(o["replicas"])["gpu_loads"] - baseline for o in polls],
                    totals(data["final"])["gpu_loads"] - baseline,
                ],
                **style,
            )
            loads = [
                np.array(
                    [
                        sum(
                            s["num_running_reqs"] + s["num_waiting_reqs"]
                            for s in state["engine"]["loads"]
                        )
                        for state in o["replicas"]
                    ]
                )
                for o in polls
            ]
            load_times, load_values = [], []
            for observation, load in zip(polls, loads, strict=True):
                when = observation["elapsed"]
                if load_times and when - load_times[-1] > 15:
                    load_times.append(when)
                    load_values.append(np.nan)
                load_times.append(when)
                load_values.append(
                    load.std() / load.mean() if load.sum() >= 288 else np.nan
                )
            axes[1, 2].plot(load_times, load_values, **style)
            mechanism_values.setdefault(family, {})[policy] = {
                "observation_time_seconds": [o["elapsed"] for o in polls],
                "cumulative_gpu_loads": [
                    totals(o["replicas"])["gpu_loads"] - baseline for o in polls
                ],
                "busy_request_count_cv": [
                    float(a.std() / a.mean()) if a.sum() >= 288 else None for a in loads
                ],
                "observation_gaps_over_15_seconds": gaps,
            }
        titles = [
            "KV token hit ratio",
            "Hot-LoRA admission ratio",
            "Mean completion time (s)",
            "Output tokens/s",
            "Cumulative GPU adapter loads",
            "Busy request-count CV",
        ]
        for ax, ylabel in zip(axes.flat, titles, strict=True):
            ax.set(xlabel="Time (s)", ylabel=ylabel)
            ax.grid(alpha=0.2)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=len(policies),
            title=title + "; predeclared trial 0",
        )
        save(fig, args.output / (family + "-timeseries"))
    write(args.output / "mechanism-plot-values.json", mechanism_values)

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), layout="constrained")
    for ax, family in zip(axes, FAMILIES, strict=True):
        for policy in policies:
            times = sorted(
                r["seconds"] for r in raw[f"confirm-{family}-r0"][policy]["requests"]
            )
            ax.plot(
                times,
                np.arange(1, len(times) + 1) / len(times),
                color=COLORS[policy],
                label=label(policy),
            )
        ax.set(
            xlabel="Request completion time (s)",
            ylabel="Fraction completed",
            title=FAMILIES[family],
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(policies))
    save(fig, args.output / "completion-cdf")


if __name__ == "__main__":
    main()
