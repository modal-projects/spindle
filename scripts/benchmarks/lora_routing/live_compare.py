"""Compare completed live RL arms with matched initial checkpoints and SDK code."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controls_plots import label
from paper_results import save
from supervise import write


METRICS = (
    "elapsed_seconds",
    "output_tokens",
    "trained_output_tokens",
    "output_tokens_per_second_including_training",
    "trained_output_tokens_per_second",
    "client_training_run_mean_seconds",
    "step_mean_seconds",
    "step_p95_seconds",
    "rollout_mean_seconds",
    "rollout_p95_seconds",
    "training_with_wait_mean_s",
    "publication_remaining_wait_mean_s",
    "group_p50_seconds",
    "group_p95_seconds",
    "sampled_groups",
    "filtered_groups",
    "unused_mixed_groups",
    "retained_groups",
    "output_length_mean",
    "output_length_p50",
    "output_length_p95",
    "truncation_fraction",
    "reward_mean",
    "first_sibling_kv_hit_ratio",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports", nargs="+", required=True, help="policy=report.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-policy-only",
        action="store_true",
        help="Explicitly allow short prefetched runs without updated-policy sampling; still require every optimizer update and publication",
    )
    args = parser.parse_args()
    reports = {}
    for item in args.reports:
        policy, path = item.split("=", 1)
        if policy in reports:
            raise ValueError("Repeated policy")
        reports[policy] = json.loads(Path(path).read_text())
    baseline = reports["kv"]
    for policy, report in reports.items():
        assert report["config"] == baseline["config"], policy
        assert report["deployment_hardware"] == baseline["deployment_hardware"], policy
        expected_routing = "naive" if policy == "kv" else policy
        assert report["routing_policy"] == expected_routing, policy
        assert report["workload_hashes"] == baseline["workload_hashes"], policy
        assert report["source_sha256"] == baseline["source_sha256"], policy
        assert report["clients"] == baseline["clients"]
        assert report["updates"] == report["clients"] * report["config"]["updates"]
        assert report["nonzero_gradient_updates"] == report["updates"]
        if not args.initial_policy_only:
            assert report["clients_sampling_updated_policies"] == report["clients"]
        assert len(report["restored_initial_states"]) == report["clients"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy, report in reports.items():
        row = dict(policy=policy, **{m: report[m] for m in METRICS})
        row.update(
            {
                "engine_" + k: v
                for k, v in report.get("engine", {}).items()
                if isinstance(v, (int, float))
            }
        )
        rows.append(row)
    with (args.output / "live-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    changes = {
        policy: {
            metric: 100 * (report[metric] / baseline[metric] - 1)
            for metric in METRICS
            if baseline[metric] != 0
        }
        for policy, report in reports.items()
        if policy != "kv"
    }
    write(
        args.output / "live-comparison.json",
        dict(
            rows=rows,
            percent_change_from_kv=changes,
            interpretation="One completed workload execution per policy. Shared-pool clients are not independent trials. Same initial checkpoints, prompts, hyperparameters, and seeds do not guarantee identical generated trajectories under different batching; token volume, filtering, lengths, and rewards are reported separately. No population confidence interval or convergence claim.",
            updates_per_arm=baseline["updates"],
            updated_policy_sampling_required=not args.initial_policy_only,
            clients_sampling_updated_policies={
                policy: report["clients_sampling_updated_policies"]
                for policy, report in reports.items()
            },
            clients_per_arm=baseline["clients"],
            config=baseline["config"],
            deployment_hardware=baseline["deployment_hardware"],
            source_sha256=baseline["source_sha256"],
        ),
    )
    plt.rcParams.update(
        {
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {p: c for p, c in zip(reports, plt.get_cmap("tab10").colors)}
    plot_values = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for policy, report in reports.items():
        begin = report["measured_start"]
        groups = sorted(report["groups"], key=lambda g: g["end"])
        times = [(g["end"] - begin) / 60 for g in groups]
        tokens = np.cumsum([sum(g["output_lengths"]) for g in groups]) / 1e6
        updates = sorted((r["time"] - begin) / 60 for r in report["update_events"])
        lengths = np.sort([n for g in groups for n in g["output_lengths"]])
        latencies = np.sort([g["end"] - g["start"] for g in groups])
        style = dict(
            label=label("stable" if policy == "domains" else policy),
            color=colors[policy],
        )
        axes[0, 0].step([0, *times], [0, *tokens], where="post", **style)
        axes[0, 1].step([0, *updates], range(len(updates) + 1), where="post", **style)
        axes[1, 0].plot(lengths, np.arange(1, len(lengths) + 1) / len(lengths), **style)
        axes[1, 1].plot(
            latencies, np.arange(1, len(latencies) + 1) / len(latencies), **style
        )
        plot_values[policy] = dict(
            group_completion_minutes=times,
            cumulative_output_millions=tokens.tolist(),
            published_update_minutes=updates,
            sorted_output_lengths=lengths.tolist(),
            sorted_sdk_group_seconds=latencies.tolist(),
        )
    labels = [
        ("Minutes from first rollout", "Generated tokens (millions)"),
        ("Minutes from first rollout", "Published optimizer updates"),
        ("Output tokens per sequence", "Fraction of sequences"),
        ("SDK group completion time (s)", "Fraction of groups"),
    ]
    for ax, (xlabel, ylabel) in zip(axes.flat, labels, strict=True):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=len(reports))
    save(fig, args.output / "live-progress")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for ax, metric, ylabel in zip(
        axes,
        ("training_run_seconds", "rollout_seconds", "train_seconds"),
        (
            "Client completion (s)",
            "Mean client rollout (s)",
            "Mean training + queue wait (s)",
        ),
        strict=True,
    ):
        for index, (policy, report) in enumerate(reports.items()):
            values = [np.mean(c[metric]) for c in report["per_client"]]
            offsets = np.linspace(-0.15, 0.15, len(values))
            ax.scatter(index + offsets, values, color=colors[policy], s=17)
            ax.plot(
                [index - 0.2, index + 0.2],
                [np.mean(values)] * 2,
                color=colors[policy],
                linewidth=2,
            )
        ax.set_xticks(
            range(len(reports)),
            [label("stable" if p == "domains" else p) for p in reports],
            rotation=25,
            ha="right",
        )
        ax.set(ylabel=ylabel, ylim=(0, None))
        ax.grid(axis="y", alpha=0.2)
    save(fig, args.output / "live-client-times")
    write(args.output / "live-plot-values.json", plot_values)


if __name__ == "__main__":
    main()
