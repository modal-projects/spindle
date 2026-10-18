"""Preserve incomplete live experiments without extrapolating completed-run rates."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from report import events, percentile
from supervise import write


def summarize(root):
    state = json.loads((root / "status.json").read_text())
    cfg = json.loads((root / "config.json").read_text())
    clients, groups, trains, rollouts, raw = [], [], [], [], []
    for folder in sorted(root.glob("client-*/")):
        trace = events(folder / "events.jsonl")
        updates = [r for r in trace if r["event"] == "update_done"]
        completed = [r for r in trace if r["event"] == "completed"]
        client_trains = [r for r in trace if r["event"] == "train_done"]
        trains.extend(client_trains)
        rollouts.extend(r for r in trace if r["event"] == "rollout_done")
        clients.append(
            dict(
                client=folder.name,
                completed=bool(completed),
                updates=len(updates),
                events=trace,
            )
        )
        for path in sorted(folder.glob("group-*.json.gz")):
            with gzip.open(path, "rt") as stream:
                group = json.load(stream)
            groups.append(
                dict(
                    client=folder.name,
                    index=group["group"],
                    start=group["start"],
                    end=group["end"],
                    policy_version=group["policy_version"],
                    prompt_tokens=len(group["prompt"]),
                    output_lengths=[len(s["tokens"]) for s in group["sequences"]],
                    rewards=[s["reward"] for s in group["sequences"]],
                )
            )
            raw.append(
                dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            )
    assert all(
        math.isfinite(float(value))
        for row in trains
        for key in ("metrics", "optimizer_metrics")
        for value in row[key].values()
    )
    lengths = [n for group in groups for n in group["output_lengths"]]
    result = dict(
        state=state["state"],
        error=state.get("error"),
        interpretation="Incomplete validation. Counts include every saved completed group, including groups from unfinished rollouts. No completed-run throughput is estimated.",
        config=cfg,
        clients=len(clients),
        completed_clients=sum(c["completed"] for c in clients),
        published_updates=sum(c["updates"] for c in clients),
        nonzero_gradient_updates=sum(
            r["optimizer_metrics"].get("grad_norm:mean", 0) > 0 for r in trains
        ),
        optimizer_successes=sum(
            r["optimizer_metrics"].get("update_successful:mean", 0) == 1 for r in trains
        ),
        completed_groups=len(groups),
        completed_sequences=len(lengths),
        completed_output_tokens=sum(lengths),
        completed_rollout_training_tokens=sum(
            r["training_input_tokens"] for r in rollouts
        ),
        output_length_mean=statistics.mean(lengths),
        output_length_p50=statistics.median(lengths),
        output_length_p95=percentile(lengths, 0.95),
        truncation_fraction=sum(n >= cfg["max_tokens"] for n in lengths) / len(lengths),
        sampled_policy_versions=sorted({g["policy_version"] for g in groups}),
        per_client=clients,
        groups=groups,
        raw_group_files=raw,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    write(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"config", "per_client", "groups", "raw_group_files"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
