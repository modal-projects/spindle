"""Client-visible reward and latency; no GPU-hardware-utilization claims."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gpu_ci.validate import events


def render(root):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for p in sorted(root.glob("client-*/events.jsonl")):
        rows = events(p)
        reward = [r for r in rows if r["event"] == "rollout_done"]
        steps = [r for r in rows if r["event"] == "update_done"]
        axes[0].plot(
            [r["update"] for r in reward],
            [r["reward"] for r in reward],
            label=p.parent.name,
        )
        axes[1].plot(
            [r["update"] for r in steps],
            [r["seconds"] for r in steps],
            label=p.parent.name,
        )
    axes[0].set(title="Training rollout reward", ylabel="Mean reward", ylim=(0, 1))
    axes[1].set(title="End-to-end update time", ylabel="Seconds")
    for ax in axes:
        ax.set_xlabel("Update")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(root / "reward-step-time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
