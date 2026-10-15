"""Retain every calibration trial, including first-use costs and startup outliers."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, folder in [
        ("1 H200 (TP1)", "validation/admission40"),
        ("2 H100 (TP2)", "validation-h100"),
    ]:
        source = args.study / folder / "calibration.json"
        data = json.loads(source.read_text())
        automatic = json.loads((source.parent / "automatic.json").read_text())
        assert data["exact_restore"]
        assert automatic["merged_requests"] == 36
        assert automatic["specialized"]["mode"]["merged_adapter"]
        assert automatic["after"]["mode"]["merged_adapter"] is None
        warm = {
            mode: statistics.mean(
                r["seconds"]
                for r in data["rows"]
                if r["mode"] == mode and r["trial"] > 0
            )
            for mode in ["unmerged", "merged"]
        }
        results[label] = dict(
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            exact_restore=True,
            automatic_merged_requests=automatic["merged_requests"],
            all_trials=data["rows"],
            transitions=data["transitions"],
            calibrated_profile=data["profile"],
            warm_mean_seconds=warm,
            warm_latency_change_percent=100 * (warm["merged"] / warm["unmerged"] - 1),
            warm_throughput_change_percent=100
            * (warm["unmerged"] / warm["merged"] - 1),
        )
    (args.output / "calibration.json").write_text(json.dumps(results, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, (label, result) in zip(axes, results.items()):
        for mode, marker in [("unmerged", "o"), ("merged", "s")]:
            rows = [r for r in result["all_trials"] if r["mode"] == mode]
            ax.plot(
                [r["trial"] for r in rows],
                [r["seconds"] for r in rows],
                marker=marker,
                label=mode,
            )
        ax.set(
            title=label,
            xlabel="Paired trial (0 includes startup)",
            ylabel="36 × 256-token batch completion (s)",
            xticks=[0, 1, 2],
        )
        ax.grid(alpha=0.25)
        ax.legend()
    for extension in ["png", "pdf"]:
        fig.savefig(args.output / ("calibration." + extension), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
