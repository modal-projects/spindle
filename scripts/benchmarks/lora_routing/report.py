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
            math.isfinite(float(v))
            for r in trains
            for kind in ("metrics", "optimizer_metrics")
            for v in r[kind].values()
        ):
            raise ValueError("Nonfinite training or optimizer metrics")
        if any(
            r["optimizer_metrics"].get("update_successful:mean", 1) != 1 for r in trains
        ):
            raise ValueError("An optimizer update was skipped")
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
                "first_rollout_at": min(rollout_starts.values()),
                "completed_at": complete[-1]["time"],
                "training_run_seconds": complete[-1]["time"]
                - min(rollout_starts.values()),
                "updates": len(updates),
                "independent_prefetch": independent,
                "nonzero_gradient_updates": sum(
                    r["optimizer_metrics"].get("grad_norm:mean", 0) > 0 for r in trains
                ),
                "sampled_policy_versions": [r["policy_version"] for r in rollouts],
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
        "measured_start": min(starts),
        "measured_end": max(ends),
        "output_tokens_per_second_including_training": tokens / elapsed,
        "throughput_measurement": "First client rollout to last client completion; includes later clients' initialization and startup staggering. Not steady-state throughput.",
        "first_rollout_spread_seconds": max(c["first_rollout_at"] for c in clients)
        - min(c["first_rollout_at"] for c in clients),
        "client_training_run_mean_seconds": statistics.mean(
            c["training_run_seconds"] for c in clients
        ),
        "step_mean_seconds": statistics.mean(all_steps),
        "step_p50_seconds": statistics.median(all_steps),
        "step_p95_seconds": percentile(all_steps, 0.95),
        "rollout_mean_seconds": statistics.mean(all_rollouts),
        "rollout_p50_seconds": statistics.median(all_rollouts),
        "rollout_p95_seconds": percentile(all_rollouts, 0.95),
        "cross_client_progress": cross_client_progress,
        "mixed_reward_groups": mixed_groups,
        "nonzero_gradient_updates": nonzero_updates,
        "clients_with_nonzero_gradients": sum(
            c["nonzero_gradient_updates"] > 0 for c in clients
        ),
        "clients_sampling_updated_policies": sum(
            max(c["sampled_policy_versions"]) > 0 for c in clients
        ),
        "group_p50_seconds": statistics.median(group_seconds),
        "group_p95_seconds": percentile(group_seconds, 0.95),
        "config": cfg,
        "workload_hashes": list(hashes)[0],
        "per_client": clients,
    }


def summarize_replicas(paths, app_name, start, end, slots=8):
    rows = sorted(
        (row for path in paths for row in events(path) if row["app"] == app_name),
        key=lambda row: row["time"],
    )
    before, after = {}, {}
    busy_seconds = overloaded_seconds = active_adapter_seconds = 0.0
    peak_requests = peak_adapters = 0
    for index, row in enumerate(rows):
        if row["time"] > end:
            break
        for replica in row["replicas"]:
            if "error" in replica:
                continue
            after[replica["replica"]] = replica["adapter_loads"]
            if row["time"] < start:
                before[replica["replica"]] = replica["adapter_loads"]
        if row["time"] < start:
            continue
        next_time = rows[index + 1]["time"] if index + 1 < len(rows) else end
        seconds = max(0, min(next_time, end) - row["time"])
        valid = [r for r in row["replicas"] if "error" not in r]
        peak_requests = max(peak_requests, sum(r["active"] for r in valid))
        for replica in valid:
            adapters = len(replica["adapters"])
            peak_adapters = max(peak_adapters, adapters)
            if replica["active"]:
                busy_seconds += seconds
                active_adapter_seconds += adapters * seconds
                overloaded_seconds += (adapters > slots) * seconds
    return {
        "adapter_registrations": sum(
            n - before.get(replica, 0) for replica, n in after.items()
        ),
        "peak_observed_active_requests": peak_requests,
        "peak_active_adapters_on_one_replica": peak_adapters,
        "mean_active_adapters_on_busy_replicas": active_adapter_seconds / busy_seconds
        if busy_seconds
        else None,
        "fraction_busy_replica_time_over_adapter_slots": overloaded_seconds
        / busy_seconds
        if busy_seconds
        else None,
        "measurement": "Active includes queued requests. Registrations are CPU loads, not GPU transfer counts.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--validate-pilot", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--require-learning", action="store_true")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--observations", nargs="*", type=Path, default=[])
    p.add_argument("--pool-app")
    args = p.parse_args()
    result = summarize(args.root, args.validate_pilot)
    if args.observations:
        if not args.pool_app:
            p.error("--observations requires --pool-app")
        result["replica_observations"] = summarize_replicas(
            args.observations,
            args.pool_app,
            result["measured_start"],
            result["measured_end"],
        )
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
