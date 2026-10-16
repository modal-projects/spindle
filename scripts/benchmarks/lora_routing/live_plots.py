"""Plot measured SDK progress and output lengths, retaining unfinished clients."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_results import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    groups = data["groups"]
    begin = min(g["start"] for g in groups)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for index, client in enumerate(data["per_client"]):
        if "events" not in client:
            continue
        rows = [r for r in client["events"] if r["event"] == "update_done"]
        axes[0].plot(
            [(r["time"] - begin) / 60 for r in rows],
            [index] * len(rows),
            "o",
            markersize=4,
            color="tab:blue" if client["completed"] else "tab:red",
        )
    axes[0].set(
        xlabel="Minutes from first measured group",
        ylabel="Independent training client",
        yticks=range(data["clients"]),
    )
    axes[0].set_title("Published optimizer updates; unfinished client in red")
    lengths = np.sort([n for g in groups for n in g["output_lengths"]])
    axes[1].plot(lengths, np.arange(1, len(lengths) + 1) / len(lengths))
    axes[1].axvline(data["config"]["max_tokens"], color="grey", linestyle="--")
    axes[1].set(
        xlabel="Generated tokens per completed sequence",
        ylabel="Empirical CDF",
        ylim=(0, 1.02),
    )
    axes[1].set_title(
        f"{len(lengths):,} completed sequences, including unfinished rollouts"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Live RL pilot: incomplete validation, not a throughput comparison")
    save(fig, args.output / "live-pilot-progress")
    values = dict(
        output_lengths=lengths.tolist(),
        update_times=[
            dict(
                client=c["client"],
                completed=c["completed"],
                updates=[
                    r["time"] - begin
                    for r in c["events"]
                    if r["event"] == "update_done"
                ],
            )
            for c in data["per_client"]
        ],
    )
    (args.output / "live-pilot-plot-values.json").write_text(json.dumps(values))


if __name__ == "__main__":
    main()
