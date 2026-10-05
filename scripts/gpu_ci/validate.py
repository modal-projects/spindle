"""Validate correctness and compare performance against a reviewed reference."""

import json
import math
import statistics

MAX_THROUGHPUT_DROP = 0.15


def validate_reference(reference, benchmark):
    require(
        reference["schema"] == 1 and reference["metric"] == "end_to_end_output_tps",
        "Unsupported reference metric/schema",
    )
    require(
        math.isfinite(reference["value"]) and reference["value"] > 0,
        "Invalid reference throughput",
    )
    require(
        reference["identity"]["run"] == benchmark,
        "Reference workload/topology mismatch; record a new reference",
    )


def compare_performance(value, identity, reference):
    validate_reference(reference, identity["run"])
    require(
        identity == reference["identity"],
        "Reference prompts/packages mismatch; record a new reference",
    )
    require(math.isfinite(value) and value > 0, "Invalid measured throughput")
    minimum = reference["value"] * (1 - MAX_THROUGHPUT_DROP)
    return dict(
        status="passed" if value >= minimum else "failed",
        metric="end_to_end_output_tps",
        reference_commit=reference["commit"],
        reference_run=reference["run_name"],
        reference_value=reference["value"],
        current_value=value,
        minimum_value=minimum,
        change_fraction=value / reference["value"] - 1,
        maximum_drop_fraction=MAX_THROUGHPUT_DROP,
    )


def events(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A supervisor can observe the live writer's final partial line.
            if index != len(lines) - 1:
                raise
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_client(rows, manifest, cfg, expected):
    require(not any(r["event"] == "failed" for r in rows), "Client reported failure")
    completed = [r for r in rows if r["event"] == "completed"]
    require(
        len(completed) == 1 and completed[0]["updates"] == cfg["updates"],
        "Incomplete client",
    )
    require(manifest["config"] == cfg, "Workload configuration changed")
    model = [r["model_id"] for r in rows if r["event"] == "client_created"]
    require(len(model) == 1, "Expected exactly one model per client")
    for event in ("rollout_done", "train_start", "train_done", "update_done"):
        records = [r for r in rows if r["event"] == event]
        require(
            sorted(r["update"] for r in records) == list(range(1, cfg["updates"] + 1)),
            f"Missing/duplicate {event}",
        )
    for r in rows:
        if "seconds" in r:
            require(
                math.isfinite(r["seconds"]) and r["seconds"] > 0, "Invalid duration"
            )
        if r["event"] == "rollout_done":
            lengths = r["output_lengths"]
            require(
                r["groups_sampled"] == expected["groups_per_update"]
                and r["groups_filtered"] == 0,
                "Group workload changed",
            )
            require(
                len(lengths)
                == expected["groups_per_update"] * expected["samples_per_group"],
                "Response count changed",
            )
            require(
                all(isinstance(n, int) and 0 < n <= cfg["max_tokens"] for n in lengths),
                "Invalid response length",
            )
            require(
                sum(lengths) == r["output_tokens"] == r["used_output_tokens"],
                "Generated/trained token mismatch",
            )
            require(0 <= r["reward"] <= 1, "Invalid reward")
        if r["event"] == "train_start":
            require(
                0 <= r["policy_lag"] <= expected["max_policy_lag"],
                "Policy lag exceeded",
            )
        if r["event"] == "train_done":
            metrics = {**r["metrics"], **r["optimizer_metrics"]}
            require(
                all(math.isfinite(float(v)) for v in metrics.values()),
                "Nonfinite training metrics",
            )
            require(
                r["optimizer_metrics"].get("update_successful:mean") == 1,
                "Optimizer did not report success",
            )
    publications = [r for r in rows if r["event"] == "publish"]
    require(
        sorted(r["update"] for r in publications) == list(range(cfg["updates"] + 1)),
        "Missing sampler publication",
    )
    require(
        len({r["path"] for r in publications}) == len(publications),
        "Sampler paths were reused",
    )
    run_id = model[0] if ":train:" in model[0] else model[0] + ":train:0"
    require(
        all(
            r["path"].startswith(f"tinker://{run_id}/sampler_weights/")
            for r in publications
        ),
        "Sampler belongs to another adapter",
    )
    probes = [r for r in rows if r["event"] == "policy_probe"]
    require(
        len(probes) == 1 and math.isfinite(probes[0]["max_logprob_change"]),
        "Missing/invalid policy probe",
    )
    # Avoid a vacuous successful run where every group has zero advantages.
    require(
        any(r["event"] == "rollout_done" and r["mixed_groups"] > 0 for r in rows),
        "No nonzero-advantage group; validation inconclusive",
    )
    require(
        any(
            r["event"] == "train_done"
            and r["optimizer_metrics"].get("grad_norm:mean", 0) > 0
            for r in rows
        ),
        "No nonzero gradient observed",
    )
    require(
        probes[0]["max_logprob_change"] > 0, "Published policy probe did not change"
    )
    return model[0]


def validate_topology(snapshot, model_ids, expected):
    placements = snapshot["placements"]
    require(
        set(placements) == set(model_ids) and all(placements.values()),
        "Missing model placement",
    )
    require(
        len({p["engine_instance_id"] for p in placements.values()}) == 1,
        "Clients did not share one trainer",
    )
    require(
        len({p["engine_boot_id"] for p in placements.values()}) == 1,
        "Trainer restarted between placements",
    )
    trainer = [
        r for r in snapshot["tasks"] if r["role"] == "trainer" and not r["finished_at"]
    ]
    inference = [
        r for r in snapshot["tasks"] if r["role"] == "lora" and not r["finished_at"]
    ]
    require(
        len(trainer) == 1 and trainer[0]["gpu_count"] == expected["trainer_gpus"],
        "Wrong trainer GPU allocation",
    )
    require(
        len(inference) == expected["inference_replicas"]
        and all(
            r["gpu_count"] == expected["inference_gpus_per_replica"] for r in inference
        ),
        "Wrong inference allocation",
    )
    require(all(r["gpu_type"] == "H200" for r in trainer + inference), "Wrong GPU type")


def report(root):
    plan = json.loads((root / "plan.json").read_text())
    cfg = json.loads((root / "config.json").read_text())
    expected = json.loads((root / "expected.json").read_text())
    paths = sorted(root.glob("client-*/events.jsonl"))
    require(len(paths) == expected["clients"], "Missing clients")
    clients = [events(p) for p in paths]
    manifests = [json.loads((p.parent / "manifest.json").read_text()) for p in paths]
    ids = [
        validate_client(rows, m, cfg, expected)
        for rows, m in zip(clients, manifests, strict=True)
    ]
    require(len(set(ids)) == expected["clients"], "Clients reused a model")
    for key in (
        "prompts_sha256",
        "dataset_sha256",
        "config_sha256",
        "stop_token_ids",
        "packages",
    ):
        require(
            len({json.dumps(m[key], sort_keys=True) for m in manifests}) == 1,
            f"Client {key} mismatch",
        )
    require(manifests[0]["dataset_sha256"] == plan["dataset_sha256"], "Dataset changed")
    if plan["mode"] == "correctness":
        last_ready = max(
            next(r["time"] for r in rows if r["event"] == "ready") for rows in clients
        )
        require(
            all(
                next(r["time"] for r in rows if r["event"] == "training_started")
                >= last_ready
                for rows in clients
            ),
            "Correctness barrier did not overlap clients",
        )
    validate_topology(json.loads((root / "topology.json").read_text()), ids, expected)
    rows = [r for client in clients for r in client]
    rollouts = [r for r in rows if r["event"] == "rollout_done"]
    tokens = sum(r["output_tokens"] for r in rollouts)
    start = min(r["time"] - r["seconds"] for r in rollouts)
    end = max(r["time"] for r in rollouts)
    summary = dict(
        status="passed",
        commit=plan["commit"],
        mode=plan["mode"],
        clients=len(clients),
        updates_per_client=cfg["updates"],
        output_tokens=tokens,
        per_client_rollout_tps=tokens / sum(r["seconds"] for r in rollouts),
        aggregate_rollout_tps=tokens / (end - start),
        reward=statistics.mean(r["reward"] for r in rollouts),
        mean_output_length=tokens / sum(len(r["output_lengths"]) for r in rollouts),
        truncation_fraction=sum(r["truncated"] for r in rollouts)
        / sum(len(r["output_lengths"]) for r in rollouts),
    )
    if plan["mode"] == "performance":
        starts = [
            next(r["time"] for r in client if r["event"] == "training_started")
            for client in clients
        ]
        finishes = [
            next(r["time"] for r in client if r["event"] == "training_finished")
            for client in clients
        ]
        require(max(finishes) > min(starts), "Invalid training measurement window")
        summary["end_to_end_output_tps"] = tokens / (max(finishes) - min(starts))
        summary["benchmark_identity"] = dict(
            run=plan["benchmark"],
            client={
                key: manifests[0][key]
                for key in ("prompts_sha256", "packages", "stop_token_ids")
            },
        )
        if plan["record_reference"]:
            summary["performance"] = {"status": "reference_candidate"}
        else:
            summary["performance"] = compare_performance(
                summary["end_to_end_output_tps"],
                summary["benchmark_identity"],
                json.loads((root / "baseline.json").read_text()),
            )
            summary["status"] = summary["performance"]["status"]
    for event in ("update_done", "rollout_done", "train_done", "publish"):
        values = [
            r["seconds"]
            for r in rows
            if r["event"] == event and 1 < r.get("update", 0) < cfg["updates"]
        ]
        summary[event] = {
            "mean_seconds": statistics.mean(values),
            "p95_seconds": sorted(values)[math.ceil(0.95 * len(values)) - 1],
            "count": len(values),
        }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# DAPO GPU CI",
        "",
        f"**{summary['status'].capitalize()}**: {plan['mode']}, {len(clients)} clients × {cfg['updates']} updates; commit `{plan['commit']}`.",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Aggregate rollout output tok/s | {summary['aggregate_rollout_tps']:,.0f} |",
        f"| Per-client rollout output tok/s | {summary['per_client_rollout_tps']:,.0f} |",
    ]
    for event in ("update_done", "rollout_done", "train_done", "publish"):
        lines.append(
            f"| {event}: middle-update mean / p95 | {summary[event]['mean_seconds']:.1f}s / {summary[event]['p95_seconds']:.1f}s |"
        )
    lines += [
        "",
        "Sampling overlaps training. Training latency includes queue/transport time. Warmup and final policy probes are excluded from these timing/token summaries, but consume resources.",
        "",
    ]
    if plan["mode"] == "performance":
        lines.append(
            f"End-to-end output throughput: **{summary['end_to_end_output_tps']:,.0f} tok/s** (includes training, publication, client ramp-up and drain; excludes initial warmup/final probes)."
        )
        comparison = summary["performance"]
        if comparison["status"] == "reference_candidate":
            lines.append(
                "Reference candidate: no performance comparison. Review and commit reference.json after successful cleanup; never replace the baseline automatically."
            )
        else:
            lines.append(
                f"Reference `{comparison['reference_commit']}`: {comparison['reference_value']:,.0f} tok/s. Minimum: {comparison['minimum_value']:,.0f} tok/s (85% of reference). Change: {comparison['change_fraction']:+.1%}. **{comparison['status']}**."
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    require(
        summary["status"] == "passed",
        "Performance regression: end-to-end throughput is more than 15% below the reference; see summary.json",
    )
    return summary
