"""Plot transport dispatch and completion for the first hot sampling group."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmarks.lora_routing.cache_report import COLORS, LABELS, STYLES


def render(roots, output, arms):
    fig, axes = plt.subplots(
        len(roots),
        2,
        figsize=(12, 3.8 * len(roots)),
        squeeze=False,
        layout="constrained",
    )
    summary = []
    for row, root in enumerate(roots):
        for arm in arms:
            data = json.loads((root / f"{arm}.json").read_text())
            requests = [
                r for r in data["requests"] if r["client"] == 0 and r["turn"] == 0
            ]
            assert len(requests) == data["config"]["hot_samples"]
            dispatch, completion = [], []
            merged = 0
            for request in requests:
                attempt = request["attempts"][0]
                trace = attempt["attrs"]["http_trace"]
                dispatch.append(trace[0]["t"] - attempt["t0"])
                completion.append(request["seconds"])
                merged += any(
                    a["attrs"].get("ok") and a["attrs"].get("lora_merged")
                    for a in request["attempts"]
                )
            summary.append(
                dict(
                    trial=str(root),
                    policy=arm,
                    requests=len(requests),
                    merged_requests=merged,
                    first_http_event_delay_mean_s=float(np.mean(dispatch)),
                    first_http_event_delay_p95_s=float(np.quantile(dispatch, 0.95)),
                    first_http_event_delay_max_s=max(dispatch),
                    request_mean_s=float(np.mean(completion)),
                    request_p95_s=float(np.quantile(completion, 0.95)),
                    request_max_s=max(completion),
                )
            )
            for column, values in enumerate([dispatch, completion]):
                axes[row, column].step(
                    sorted(values),
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    label=LABELS[arm],
                    color=COLORS[arm],
                    linestyle=STYLES[arm],
                )
            del data, requests
        axes[row, 0].set_xscale("symlog", linthresh=0.1)
        for column, label in enumerate(
            ["Delay to first HTTP transport event (s)", "Request completion time (s)"]
        ):
            axes[row, column].set(
                xlabel=label,
                ylabel="Fraction of hot-group requests",
                ylim=(0, 1.02),
                title=root.parent.name,
            )
            axes[row, column].grid(alpha=0.25)
        axes[row, 1].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix("." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms", nargs="+", default=["kv", "model-spill", "dlora-no-merge", "dlora"]
    )
    args = parser.parse_args()
    render(args.roots, args.output, args.arms)
