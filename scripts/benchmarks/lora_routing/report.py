"""Summarize independent client latencies; validate a two-step pilot."""

import argparse
import gzip
import json
import math
from pathlib import Path
import statistics


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def summarize(root, require_independence=False):
    clients = []
    all_steps = []
    all_rollouts = []
    all_groups = []
    starts = []
    ends = []
    group_seconds = []
    mixed_groups = 0
    nonzero_updates = 0
    model_ids = set()
    hashes = set()
    timelines = []
    for folder in sorted(root.glob("client-*")):
        if not folder.is_dir():
            continue
        trace = events(folder / "events.jsonl")
        manifest = json.loads((folder / "manifest.json").read_text())
        hashes.add(
            (
                manifest["config_sha256"],
                manifest["prompts_sha256"],
                manifest["dataset_sha256"],
            )
        )
        model_ids.update(r["model_id"] for r in trace if r["event"] == "client_created")
        for group_file in folder.glob("group-*.json.gz"):
            with gzip.open(group_file, "rt") as stream:
                group = json.load(stream)
            group_seconds.append(group["end"] - group["start"])
            mixed_groups += len({r["reward"] for r in group["sequences"]}) > 1
        timelines.append(trace)
        complete = [r for r in trace if r["event"] == "completed"]
        if not complete:
            raise ValueError(f"{folder.name} did not complete")
        rollouts = [r for r in trace if r["event"] == "rollout_done"]
        updates = [r for r in trace if r["event"] == "update_done"]
        trains = [r for r in trace if r["event"] == "train_done"]
        if len(updates) != manifest["config"]["updates"]:
            raise ValueError("Missing updates")
        if not all(
            math.isfinite(float(v)) for r in trains for v in r["metrics"].values()
        ):
            raise ValueError("Nonfinite training metrics")
        rollout_starts = {
            r["update"]: r["time"] for r in trace if r["event"] == "rollout_start"
        }
        nonzero_updates += sum(
            r["optimizer_metrics"].get("grad_norm:mean", 0) > 0 for r in trains
        )
        independent = any(
            rollout_starts.get(r["update"] + 1, math.inf) < r["time"] for r in trains
        )
        if require_independence and not independent:
            raise ValueError("No independent rollout/train overlap")
        steps = [r["time"] - rollout_starts[r["update"]] for r in updates]
        all_steps.extend(steps)
        all_rollouts.extend(r["seconds"] for r in rollouts)
        starts.extend(rollout_starts.values())
        ends.append(complete[-1]["time"])
        all_groups.extend(rollouts)
        clients.append(
            {
                "client": folder.name,
                "updates": len(updates),
                "independent_prefetch": independent,
                "step_seconds": steps,
                "rollout_seconds": [r["seconds"] for r in rollouts],
                "train_seconds": [r["seconds"] for r in trains],
                "publication_seconds": [
                    r["seconds"] for r in trace if r["event"] == "publish"
                ],
                "reward": [r["reward"] for r in rollouts],
                "optimizer_metrics": [r["optimizer_metrics"] for r in trains],
            }
        )
    if len(hashes) != 1:
        raise ValueError("Client workload hashes differ")
    if len(model_ids) != len(clients):
        raise ValueError("Clients did not use distinct trainable adapters")
    cfg = json.loads((root / "config.json").read_text())
    tokens = sum(r["output_tokens"] for r in all_groups)
    elapsed = max(ends) - min(starts)
    cross_client_progress = any(
        start["event"] == "rollout_start"
        and start["update"] > 1
        and any(
            end["event"] == "update_done"
            and end["update"] == start["update"] - 1
            and start["time"] < end["time"]
            for end in other
        )
        for i, trace in enumerate(timelines)
        for start in trace
        for j, other in enumerate(timelines)
        if i != j
    )
    if require_independence and len(clients) > 1 and not cross_client_progress:
        raise ValueError("No progress across clients before a shared step boundary")
    return {
        "clients": len(clients),
        "updates": len(all_steps),
        "output_tokens": tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second_including_training": tokens / elapsed,
        "step_mean_seconds": statistics.mean(all_steps),
        "step_p50_seconds": statistics.median(all_steps),
        "step_p95_seconds": percentile(all_steps, 0.95),
        "rollout_mean_seconds": statistics.mean(all_rollouts),
        "rollout_p50_seconds": statistics.median(all_rollouts),
        "rollout_p95_seconds": percentile(all_rollouts, 0.95),
        "cross_client_progress": cross_client_progress,
        "mixed_reward_groups": mixed_groups,
        "nonzero_gradient_updates": nonzero_updates,
        "group_p50_seconds": statistics.median(group_seconds),
        "group_p95_seconds": percentile(group_seconds, 0.95),
        "config": cfg,
        "workload_hashes": list(hashes)[0],
        "per_client": clients,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--validate-pilot", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--require-learning", action="store_true")
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    result = summarize(args.root, args.validate_pilot)
    if args.require_learning and not (
        result["mixed_reward_groups"] and result["nonzero_gradient_updates"]
    ):
        raise ValueError("No mixed rewards and nonzero policy gradients")
    if args.baseline:
        baseline = summarize(args.baseline)
        if (baseline["workload_hashes"], baseline["clients"]) != (
            result["workload_hashes"],
            result["clients"],
        ):
            raise ValueError("Baseline workload differs")
        result["baseline_ratios"] = {
            key: result[key] / baseline[key]
            for key in (
                "step_mean_seconds",
                "step_p95_seconds",
                "rollout_mean_seconds",
                "rollout_p95_seconds",
                "group_p95_seconds",
                "output_tokens_per_second_including_training",
            )
        }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
