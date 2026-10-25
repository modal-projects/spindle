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
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
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
    ]
    for axis, (label, values) in zip(axes.flat, series, strict=True):
        axis.plot(steps, values, linewidth=1.3)
        axis.set(xlabel="Optimizer update", title=label)
        axis.grid(alpha=0.2)
    figure.suptitle(f"Asynchronous mixed-sampler RL — {len(updates)} measured updates")
    figure.savefig(args.run / "training-curves.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    groups = sorted({g for e in updates for g in e["groups"]})
    for group in groups:
        selected = [e for e in updates if group in e["groups"]]
        xs = [e["step"] for e in selected]
        axes[0].plot(xs, [e["groups"][group]["reward"] for e in selected], label=group)
        axes[1].plot(
            xs, [e["groups"][group]["policy_age_mean"] for e in selected], label=group
        )
    axes[0].set(title="Test pass rate by sampler group", xlabel="Optimizer update")
    axes[1].set(
        title="Mean policy age by sampler group",
        xlabel="Optimizer update",
        ylabel="Publications behind trainer",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    figure.savefig(args.run / "sampler-groups.png", dpi=160)
    plt.close(figure)
    print(f"Wrote curves for {len(updates)} updates to {args.run}")


if __name__ == "__main__":
    main()
