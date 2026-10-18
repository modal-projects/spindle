"""Validate actual SDK optimization and retain lengths, filtering, and queue costs."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from report import events, percentile, summarize
from supervise import write


def summarize_live(root, observations=None):
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "completed", status
    result = summarize(root, require_independence=True)
    cfg = result["config"]
    assert result["clients"] == status["clients"]
    assert result["nonzero_gradient_updates"] == result["updates"]
    if cfg["updates"] >= 3:
        assert result["clients_sampling_updated_policies"] == result["clients"]
    groups, rollouts, traces, files, update_events, restored_states = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    source_hashes = set()
    for folder in sorted(root.glob("client-*/")):
        trace = events(folder / "events.jsonl")
        manifest = json.loads((folder / "manifest.json").read_text())
        source_hashes.add(tuple(sorted(manifest["source_sha256"].items())))
        if cfg.get("initial_state_paths"):
            restored = [r for r in trace if r["event"] == "initial_state_loaded"]
            expected = cfg["initial_state_paths"][manifest["client_index"]]
            assert len(restored) == 1 and restored[0]["path"] == expected
            restored_states.append(dict(client=folder.name, **restored[0]))
        update_events.extend(
            dict(client=folder.name, update=r["update"], time=r["time"])
            for r in trace
            if r["event"] == "update_done"
        )
        traces.append(trace)
        rollouts.extend(r for r in trace if r["event"] == "rollout_done")
        for path in sorted(folder.glob("group-*.json.gz")):
            with gzip.open(path, "rt") as stream:
                group = json.load(stream)
            groups.append(
                {
                    "client": folder.name,
                    "group": group["group"],
                    "row": group["row"],
                    "policy_version": group["policy_version"],
                    "start": group["start"],
                    "end": group["end"],
                    "prompt_tokens": len(group["prompt"]),
                    "first_sequence_cached_tokens": group[
                        "first_sequence_cached_tokens"
                    ],
                    "output_lengths": [len(s["tokens"]) for s in group["sequences"]],
                    "rewards": [s["reward"] for s in group["sequences"]],
                    "stop_reasons": [s["stop_reason"] for s in group["sequences"]],
                }
            )
            files.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    lengths = [n for g in groups for n in g["output_lengths"]]
    assert len(source_hashes) == 1
    assert sum(lengths) == result["output_tokens"]
    assert all(0 < n <= cfg["max_tokens"] for n in lengths)
    trained = sum(r["used_output_tokens"] for r in rollouts)
    train_seconds = [
        r["seconds"] for trace in traces for r in trace if r["event"] == "train_done"
    ]
    publish_seconds = [
        r["seconds"]
        for trace in traces
        for r in trace
        if r["event"] == "publish" and r["update"] > 0
    ]
    result.update(
        source_sha256=dict(next(iter(source_hashes))),
        restored_initial_states=restored_states,
        update_events=update_events,
        sequences=len(lengths),
        trained_output_tokens=trained,
        trained_output_tokens_per_second=trained / result["elapsed_seconds"],
        training_input_tokens=sum(r["training_input_tokens"] for r in rollouts),
        output_length_mean=statistics.mean(lengths),
        output_length_p50=statistics.median(lengths),
        output_length_p95=percentile(lengths, 0.95),
        output_length_max=max(lengths),
        truncated_sequences=sum(n >= cfg["max_tokens"] for n in lengths),
        truncation_fraction=sum(n >= cfg["max_tokens"] for n in lengths) / len(lengths),
        reward_mean=statistics.mean(r for g in groups for r in g["rewards"]),
        sampled_groups=len(groups),
        filtered_groups=sum(r["groups_filtered"] for r in rollouts),
        unused_mixed_groups=sum(r.get("groups_unused", 0) for r in rollouts),
        retained_groups=sum(
            r["groups_sampled"] - r["groups_filtered"] - r.get("groups_unused", 0)
            for r in rollouts
        ),
        training_with_wait_mean_s=statistics.mean(train_seconds),
        publication_remaining_wait_mean_s=statistics.mean(publish_seconds),
        first_sibling_kv_hit_ratio=sum(
            g["first_sequence_cached_tokens"] for g in groups
        )
        / sum(g["prompt_tokens"] for g in groups),
        group_measurement="SDK group latency includes all sibling completions, control-plane delivery, and result retrieval. It is not individual-sequence latency.",
        kv_measurement="The SDK exposes cached tokens for the first sibling only. Engine counters below count actual reuse at first admission for all observed sequences.",
        raw_group_files=files,
        groups=groups,
    )
    if observations:
        complete = [
            r
            for r in events(observations)
            if len(r.get("replicas", [])) == 8
            and all("error" not in s for s in r["replicas"])
        ]
        assert complete[0]["time"] < result["measured_start"]
        assert complete[-1]["time"] >= result["measured_end"]
        before = {s["replica"]: s for s in complete[0]["replicas"]}
        after = {s["replica"]: s for s in complete[-1]["replicas"]}
        assert before.keys() == after.keys()
        assert all(s["active"] == 0 for s in after.values())
        measured = [
            r
            for r in complete
            if result["measured_start"] <= r["time"] <= result["measured_end"]
        ]
        gaps = [b["time"] - a["time"] for a, b in zip(measured, measured[1:])]
        observed_seconds = sum(gap for gap in gaps if gap <= 15)
        queued_seconds = unused_slots = load_cv = 0.0
        for row, gap in zip(measured, gaps):
            if gap > 15:
                continue
            loads = [
                load
                for replica in row["replicas"]
                for load in replica["engine"]["loads"]
            ]
            if not any(load["num_waiting_reqs"] for load in loads):
                continue
            queued_seconds += gap
            total_capacity = sum(load["max_running_requests"] for load in loads)
            unused_slots += (
                gap
                * sum(
                    max(0, load["max_running_requests"] - load["num_running_reqs"])
                    for load in loads
                )
                / total_capacity
            )
            requests = [
                load["num_running_reqs"] + load["num_waiting_reqs"] for load in loads
            ]
            load_cv += gap * statistics.pstdev(requests) / statistics.mean(requests)
        result["observations"] = dict(
            complete_samples=len(measured),
            max_gap_seconds=max(gaps, default=0),
            observed_seconds=observed_seconds,
            coverage_fraction=observed_seconds / result["elapsed_seconds"],
            seconds_with_any_native_queue=queued_seconds,
            unused_sequence_capacity_fraction_while_queued=unused_slots / queued_seconds
            if queued_seconds
            else None,
            request_load_cv_while_queued=load_cv / queued_seconds
            if queued_seconds
            else None,
            note="Time-weighted native scheduler observations; gaps over 15 seconds are excluded, not interpolated. Unused sequence slots do not prove usable GPU compute capacity.",
        )
        counters = {}
        for key in (
            "gpu_loads",
            "gpu_evictions",
            "load_host_seconds",
            "request_lora_hits",
            "request_lora_misses",
            "first_admission_input_tokens",
            "first_admission_cached_tokens",
        ):
            delta = sum(
                after[h]["gpu_lora"].get(key, 0) - before[h]["gpu_lora"].get(key, 0)
                for h in before
            )
            assert delta >= 0
            counters[key] = delta
        result["engine"] = dict(
            **counters,
            kv_hit_ratio=counters["first_admission_cached_tokens"]
            / counters["first_admission_input_tokens"],
            hot_lora_hit_ratio=counters["request_lora_hits"]
            / (counters["request_lora_hits"] + counters["request_lora_misses"]),
            window_start=complete[0]["time"],
            window_end=complete[-1]["time"],
            measurement="Complete observation window, including sampling warmup and later client initialization. Cache counts are recorded once at first native prefill admission; adapter load time is host elapsed time, not isolated CUDA time.",
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_live(args.root, args.observations)
    if args.deployment:
        deployment = json.loads(args.deployment.read_text())
        recipe = deployment["recipe"]
        result["routing_policy"] = recipe["inference_lora_routing"]
        result["deployment_hardware"] = {
            name: recipe[name]
            for name in (
                "model",
                "max_context_length",
                "trainer_gpu",
                "trainer_gpus_per_node",
                "trainer_nodes",
                "trainer_cpu",
                "trainer_memory_mib",
                "trainer_max_clients_per_instance",
                "inference_gpu",
                "inference_gpus_per_node",
                "inference_cpu",
                "inference_memory_mib",
                "inference_min_replicas",
                "inference_max_replicas",
                "inference_target_concurrency",
                "inference_http2",
                "miles_cfg",
                "sglang_cfg",
            )
        }
    write(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"per_client", "groups", "raw_group_files", "config"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
