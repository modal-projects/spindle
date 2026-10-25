"""Plot measured training dynamics without inferring stability from a short check."""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt

plt.switch_backend("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    events = [
        json.loads(line)
        for line in (args.run / "events.jsonl").read_text().splitlines()
    ]
    updates = [e for e in events if e["event"] == "training_completed"]
    if not updates:
        raise ValueError(
            "No completed training updates; there are no stability curves to plot"
        )
    steps = [e["step"] for e in updates]
    aggregation = sorted(
        {
            kind
            for update in updates
            for kind in update.get("diagnostic_aggregation", ["sdk_chunk_means"])
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    series = [
        ("Test pass rate", [e["reward"] for e in updates]),
        (
            "Sampler/trainer KL on top-p support",
            [e["metrics"]["sampler_trainer_kl:mean"] for e in updates],
        ),
        (
            "Gradient norm before clipping",
            [e["optimizer"]["grad_norm:mean"] for e in updates],
        ),
        (
            "Entropy on replayed support",
            [e["metrics"]["entropy:mean"] for e in updates],
        ),
        (
            "Generated tokens / trajectory",
            [e["mean_generated_tokens"] for e in updates],
        ),
        ("Tool format error rate", [e["format_error_rate"] for e in updates]),
        (
            "MIS masked token fraction",
            [e["metrics"]["mis_masked_fraction:mean"] for e in updates],
        ),
        (
            "Maximum policy age",
            [max(g["policy_age_max"] for g in e["groups"].values()) for e in updates],
        ),
    ]
    for axis, (label, values) in zip(axes.flat, series, strict=True):
        axis.plot(steps, values, linewidth=1.3, marker="o", markersize=3)
        axis.set(xlabel="Optimizer update", title=label)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Asynchronous mixed-sampler RL — {len(updates)} measured updates\n"
        f"Loss diagnostics: {', '.join(aggregation)}"
    )
    figure.savefig(args.run / "training-curves.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    groups = sorted({g for e in updates for g in e["groups"]})
    for group in groups:
        selected = [e for e in updates if group in e["groups"]]
        xs = [e["step"] for e in selected]
        axes[0].plot(
            xs,
            [e["groups"][group]["reward"] for e in selected],
            label=group,
            marker="o",
            markersize=3,
        )
        axes[1].plot(
            xs,
            [e["groups"][group]["policy_age_mean"] for e in selected],
            label=group,
            marker="o",
            markersize=3,
        )
    axes[0].set(title="Test pass rate by sampler group", xlabel="Optimizer update")
    axes[1].set(
        title="Mean policy age by sampler group",
        xlabel="Optimizer update",
        ylabel="Optimizer updates behind trainer",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    figure.savefig(args.run / "sampler-groups.png", dpi=160)
    plt.close(figure)
    starts = {e["step"]: e["at"] for e in events if e["event"] == "training_started"}
    overlaps = []
    for update in updates:
        during = [e for e in events if starts[update["step"]] < e["at"] < update["at"]]
        overlaps.append(
            {
                "step": update["step"],
                "prompts_started_during_training": sum(
                    e["event"] == "prompt_started" for e in during
                ),
                "trajectories_started_during_training": sum(
                    e["event"] == "trajectory_started" for e in during
                ),
                "trajectories_completed_during_training": sum(
                    e["event"] == "trajectory_completed" for e in during
                ),
            }
        )
    summary = {
        "completed_updates": len(updates),
        "diagnostic_aggregation": aggregation,
        "first_update": updates[0],
        "last_update": updates[-1],
        "async_overlap": overlaps,
        "trajectory_failures": [e for e in events if e["event"] == "trajectory_failed"],
        "interpretation": (
            "Measured curves and overlap evidence; finite updates alone do not establish "
            "stable learning. Inspect reward, length, entropy, KL and format-error trends, "
            "including differences between sampler groups and policy ages."
        ),
    }
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote curves for {len(updates)} updates to {args.run}")


if __name__ == "__main__":
    main()
