"""Compare independently progressing Miles clients using measured events."""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt

plt.switch_backend("Agg")


def read_events(path):
    # A live writer may not have finished the final line yet.
    lines = path.read_bytes().split(b"\n")
    return [json.loads(line) for line in lines[:-1] if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    clients = {
        index: read_events(args.run / f"client-{index}" / "events.jsonl")
        for index in range(manifest["clients"])
    }
    updates = {
        index: [e for e in rows if e["event"] == "training_completed"]
        for index, rows in clients.items()
    }
    if not any(updates.values()):
        raise ValueError("No completed Miles updates; stability curves are unavailable")
    client_events = read_events(args.run / "clients.jsonl")
    starts = {
        e["client"]: e["at"]
        for e in client_events
        if e["event"] == "client_training_started"
    }
    origin = min(starts.values())
    windows = []
    summary = {
        "run": str(args.run),
        "start_offsets_seconds": {i: t - origin for i, t in starts.items()},
        "clients": {},
        "interpretation": (
            "Measured client timelines, not simultaneous GPU execution. Restored "
            "trajectories are excluded from fresh overlap counts. Finite updates "
            "alone do not establish stable learning."
        ),
    }
    series = [
        ("Test pass rate", lambda e: e["reward"]),
        (
            "Sampler/trainer KL on top-p support",
            lambda e: e["metrics"]["sampler_trainer_kl:mean"],
        ),
        ("Gradient norm before clipping", lambda e: e["optimizer"]["grad_norm:mean"]),
        ("Entropy on replayed support", lambda e: e["metrics"]["entropy:mean"]),
        ("Generated tokens / trajectory", lambda e: e["mean_generated_tokens"]),
        ("Tool format error rate", lambda e: e["format_error_rate"]),
        (
            "MIS masked token fraction",
            lambda e: e["metrics"]["mis_masked_fraction:mean"],
        ),
        (
            "Maximum policy age (optimizer updates)",
            lambda e: max(g["policy_age_max"] for g in e["groups"].values()),
        ),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    for index, completed in updates.items():
        if not completed:
            continue
        for axis, (label, value) in zip(axes.flat, series, strict=True):
            axis.plot(
                [e["step"] for e in completed],
                [value(e) for e in completed],
                marker="o",
                markersize=3,
                label=f"Client {index}",
            )
            axis.set(xlabel="Client optimizer update", title=label)
            axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("Miles multi-LoRA — independent asynchronous clients")
    figure.savefig(args.run / "multi-client-curves.png", dpi=160)
    plt.close(figure)

    for index, rows in clients.items():
        training_starts = {
            e["step"]: e["at"] for e in rows if e["event"] == "training_started"
        }
        overlap = []
        for update in updates[index]:
            start = training_starts[update["step"]]
            end = update["at"]
            publication_start = end - update["publication_seconds"]
            windows.append((index, update["step"], start, publication_start, end))
            counts = {}
            for other, other_rows in clients.items():
                during = [e for e in other_rows if start < e["at"] < end]
                counts[other] = {
                    "fresh_trajectories_started": sum(
                        e["event"] == "trajectory_started" for e in during
                    ),
                    "fresh_trajectories_completed": sum(
                        e["event"] == "trajectory_completed" for e in during
                    ),
                    "training_steps_started": [
                        e["step"] for e in during if e["event"] == "training_started"
                    ],
                    "optimizer_updates_completed": [
                        e["step"] for e in during if e["event"] == "training_completed"
                    ],
                }
            overlap.append({"step": update["step"], "during_update_by_client": counts})
        summary["clients"][index] = {
            "completed_updates": len(updates[index]),
            "latest_update": updates[index][-1] if updates[index] else None,
            "async_overlap": overlap,
        }

    figure, axes = plt.subplots(
        2, 1, figsize=(12, 7), constrained_layout=True, sharex=True
    )
    for index, rows in clients.items():
        fresh = [e["at"] for e in rows if e["event"] == "trajectory_completed"]
        axes[0].step(
            [(t - origin) / 3600 for t in [starts[index], *fresh]],
            list(range(len(fresh) + 1)),
            where="post",
            linestyle=["-", "--", ":", "-."][index % 4],
            label=f"Client {index}",
        )
    for index, step, start, publication_start, end in windows:
        for left, right, color in [
            (start, publication_start, "tab:blue"),
            (publication_start, end, "tab:orange"),
        ]:
            axes[1].broken_barh(
                [((left - origin) / 3600, (right - left) / 3600)],
                (index - 0.3, 0.6),
                facecolors=color,
            )
        axes[1].text(
            (start + end - 2 * origin) / 7200,
            index,
            str(step),
            ha="center",
            va="center",
        )
    axes[0].set(title="Fresh rollout completions", ylabel="Trajectories")
    axes[0].legend()
    axes[1].set(
        title="Completed update intervals: training/optimizer (blue), publication (orange)",
        yticks=list(clients),
        yticklabels=[f"Client {i}" for i in clients],
    )
    for axis in axes:
        axis.set_xlabel("Hours since first client started")
        axis.grid(alpha=0.2)
    figure.savefig(args.run / "multi-client-timeline.png", dpi=160)
    plt.close(figure)
    (args.run / "multi-client-summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"Wrote comparison for {sum(map(len, updates.values()))} measured Miles updates"
    )


if __name__ == "__main__":
    main()
