"""Validate raw cache experiments and produce quantitative tables and PNG/PDF plots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"default": "Default Flash", "lora": "LoRA-aware", "kv": "KV affinity"}
LABELS.update({"work": "Measured work", "work-no-adapter": "Work without adapter cost"})
COLORS = {"default": "#d84b43", "lora": "#287fba", "kv": "#15947a"}
COLORS.update({"work": "#6a48af", "work-no-adapter": "#d89519"})
STYLES = {"default": "--", "lora": "-", "kv": "-."}
STYLES.update({"work": "-", "work-no-adapter": ":"})
COUNTERS = (
    "request_lora_hits",
    "request_lora_misses",
    "gpu_loads",
    "gpu_evictions",
    "load_host_seconds",
)


def totals(states):
    return {k: sum(s["gpu_lora"].get(k, 0) for s in states) for k in COUNTERS}


def engine_intervals(data):
    """Approximate successful-attempt occupancy; not native queue samples.

    Anchor engine e2e duration at response-sent time. SGLang records e2e finish
    just before response sending, so this shifts the interval by that gap.
    Failed/retried attempts without a response cannot be reconstructed.
    """
    hosts = sorted(s["replica"] for s in data["baseline"])
    adapters = sorted({r["adapter_index"] for r in data["requests"]})
    times = np.arange(0, data["seconds"], 1.0)
    changes = np.zeros((len(hosts), len(adapters), len(times) + 1), dtype=int)
    for row in data["requests"]:
        meta = next(
            a["attrs"]["backend_timing"]
            for a in reversed(row["attempts"])
            if a["attrs"].get("ok")
        )
        end = meta["response_sent_to_client_ts"] - data["wall_started"]
        start = end - meta["e2e_latency"]
        lo, hi = np.searchsorted(times, [start, end])
        h, a = (
            hosts.index(row["stats"]["upstream"]),
            adapters.index(row["adapter_index"]),
        )
        changes[h, a, lo] += 1
        changes[h, a, hi] -= 1
    occupancy = np.cumsum(changes, axis=2)[:, :, :-1]
    return times, occupancy.sum(axis=1), (occupancy > 0).sum(axis=1)


def summarize(data):
    assert data.get("error") is None, data.get("error")
    rows, config = data["requests"], data["config"]
    training = config.get("workload") == "dapo"
    expected = (
        config["expected_requests"]
        if training
        else 32 * config["streams"] * config["turns"]
    )
    if config.get("duration_seconds"):
        expected = len(rows)
        assert expected > 0
        assert len(data["frontends"]) == len(set(data["frontends"])) == 8
    assert len(rows) == len({r["id"] for r in rows}) == expected
    assert all(
        (training or r["prompt_tokens"] == config["prompt_tokens"])
        and r["decode_tokens"] == config["decode_tokens"]
        for r in rows
    )
    assert {s["replica"] for s in data["baseline"]} == {
        s["replica"] for s in data["final"]
    }
    assert len(data["baseline"]) == 8
    assert all(s["active"] == 0 for s in data["final"])
    assert all(
        load["num_running_reqs"] == load["num_waiting_reqs"] == 0
        for s in data["final"]
        for load in s["engine"]["loads"]
    )
    assert not any(
        a["attrs"].get("route_mismatch") for r in rows for a in r["attempts"]
    )
    before, after = totals(data["baseline"]), totals(data["final"])
    counters = {k: after[k] - before[k] for k in COUNTERS}
    assert all(v >= 0 for v in counters.values())
    admissions = counters["request_lora_hits"] + counters["request_lora_misses"]
    attempts = sum(len(r["attempts"]) for r in rows)
    assert attempts == expected + sum(r["stats"]["reroutes"] for r in rows)
    # A transport timeout can happen after engine admission. Keep this duplicate
    # work in the comparison, bounded by the recorded physical attempts, rather
    # than treating all native admissions as distinct logical requests.
    assert expected <= admissions <= attempts, (expected, admissions, attempts)
    assert counters["gpu_loads"] - counters["gpu_evictions"] == sum(
        len(s["gpu_lora"]["resident_adapters"]) for s in data["final"]
    ) - sum(len(s["gpu_lora"]["resident_adapters"]) for s in data["baseline"])
    latencies = [r["seconds"] for r in rows]
    backend_timings = [
        next(
            a["attrs"]["backend_timing"]
            for a in reversed(r["attempts"])
            if a["attrs"].get("http_status") == 200
        )
        for r in rows
    ]
    engine_latencies = [t["e2e_latency"] for t in backend_timings]
    cached, prompt = (
        sum(r["cached_tokens"] for r in rows),
        sum(r["prompt_tokens"] for r in rows),
    )
    repeats = [r for r in rows if r["turn"] > 0]
    warm = sum(r["cached_tokens"] for r in repeats) / max(
        1, sum(r["prompt_tokens"] for r in repeats)
    )
    registrations = sum(s["adapter_loads"] for s in data["final"]) - sum(
        s["adapter_loads"] for s in data["baseline"]
    )
    if not config.get("cold_versions"):
        assert registrations == 0, "Registration I/O contaminated the cache comparison"
    snapshots = [
        s for row in data["observations"] if "replicas" in row for s in row["replicas"]
    ]
    observation_times = (
        [0.0]
        + [o["elapsed"] for o in data["observations"] if "replicas" in o]
        + [data["seconds"]]
    )
    observation_gaps = [b - a for a, b in zip(observation_times, observation_times[1:])]
    capacity = sorted(
        {
            load["max_running_requests"]
            for s in snapshots
            for load in s["engine"]["loads"]
        }
    )
    max_queue = max(
        (
            sum(
                load["num_waiting_reqs"]
                for s in o["replicas"]
                for load in s["engine"]["loads"]
            )
            for o in data["observations"]
            if "replicas" in o
        ),
        default=0,
    )
    loaded = [
        o
        for o in data["observations"]
        if "replicas" in o
        and sum(s["active"] for s in o["replicas"])
        >= (config["max_concurrency"] if training else expected / config["turns"]) / 2
    ]
    load_cv = [
        float(
            np.std([s["active"] for s in o["replicas"]])
            / np.mean([s["active"] for s in o["replicas"]])
        )
        for o in loaded
    ]
    active_sets = [len(s["adapters"]) for o in loaded for s in o["replicas"]]
    # Use actual engine capacity to evaluate busy intervals. The independent
    # clients need not all overlap, so half the theoretical launch ceiling can
    # otherwise select no observations at all.
    running_capacity = sum(
        load["max_running_requests"]
        for state in data["baseline"]
        for load in state["engine"]["loads"]
    )
    busy = [
        o
        for o in data["observations"]
        if "replicas" in o
        and sum(s["active"] for s in o["replicas"]) >= running_capacity
    ]
    _, estimated_loads, estimated_sets = engine_intervals(data)
    engine_busy = estimated_loads.sum(axis=0) >= running_capacity
    busy_loads = [np.array([s["active"] for s in o["replicas"]]) for o in busy]
    busy_sets = [len(s["adapters"]) for o in busy for s in o["replicas"]]
    placements = {
        i: {r["stats"]["upstream"] for r in rows if r["adapter_index"] == i}
        for i in {r["adapter_index"] for r in rows}
    }
    prefix_placements = {}
    for row in rows:
        key = (row["adapter_index"], row.get("conversation", row["client"]))
        prefix_placements.setdefault(key, set()).add(row["stats"]["upstream"])
    lookups = data.get("lookup_metrics", [])
    lookup_count = sum(m["count"] for m in lookups)
    window_end = config.get("duration_seconds", 0)
    window_start = min(60, window_end / 3)
    window = [r for r in rows if window_start <= r["end"] < window_end]
    selections = [
        a["attrs"]["routing"]["selection_seconds"]
        for r in rows
        for a in r["attempts"]
        if a["attrs"].get("routing")
    ]
    snapshot_times = [s["work"]["snapshot_seconds"] for s in snapshots if s.get("work")]
    return {
        "requests": expected,
        "physical_attempts": attempts,
        "engine_admissions": admissions,
        "extra_engine_admissions": admissions - expected,
        "attempts_without_engine_admission": attempts - admissions,
        "seconds": data["seconds"],
        "measurement_window": [window_start, window_end] if window_end else None,
        "steady_completed_requests": len(window) if window_end else None,
        "steady_output_tokens_per_second": sum(r["decode_tokens"] for r in window)
        / (window_end - window_start)
        if window_end
        else None,
        "steady_latency_mean_s": float(np.mean([r["seconds"] for r in window]))
        if window
        else None,
        "steady_kv_hit_ratio": sum(r["cached_tokens"] for r in window)
        / max(1, sum(r["prompt_tokens"] for r in window))
        if window
        else None,
        "routing_selection_p50_us": float(np.quantile(selections, 0.5) * 1e6)
        if selections
        else None,
        "routing_selection_p95_us": float(np.quantile(selections, 0.95) * 1e6)
        if selections
        else None,
        "snapshot_p95_ms": float(np.quantile(snapshot_times, 0.95) * 1e3)
        if snapshot_times
        else None,
        "output_tokens": expected * config["decode_tokens"],
        "tokens_per_second": expected * config["decode_tokens"] / data["seconds"],
        "requests_per_second": expected / data["seconds"],
        "latency_mean_s": float(np.mean(latencies)),
        "latency_p50_s": float(np.quantile(latencies, 0.5)),
        "latency_p95_s": float(np.quantile(latencies, 0.95)),
        "latency_p99_s": float(np.quantile(latencies, 0.99)),
        "latency_max_s": max(latencies),
        "engine_e2e_mean_s": float(np.mean(engine_latencies)),
        "outside_engine_mean_s": float(
            np.mean(
                [total - engine for total, engine in zip(latencies, engine_latencies)]
            )
        ),
        "engine_retractions": sum(t["num_retractions"] for t in backend_timings),
        "cached_tokens": cached,
        "prompt_tokens": prompt,
        "kv_hit_ratio": cached / prompt,
        "repeat_kv_hit_ratio": warm,
        "lora_hit_ratio": counters["request_lora_hits"] / admissions,
        "registration_calls": registrations,
        "reroutes": sum(r["stats"]["reroutes"] for r in rows),
        "explicit_replica_attempt_fraction": (
            sum(
                bool(a["attrs"].get("selected_replica"))
                for r in rows
                for a in r["attempts"]
            )
            / attempts
            if all(
                "selected_replica" in a["attrs"] for r in rows for a in r["attempts"]
            )
            else None
        ),
        "observed_max_queued_requests": max_queue,
        "complete_snapshot_polls": sum("replicas" in o for o in data["observations"]),
        "failed_snapshot_polls": sum("error" in o for o in data["observations"]),
        "max_complete_observation_gap_s": max(observation_gaps),
        "observation_coverage_10s": sum(min(10, gap) for gap in observation_gaps)
        / data["seconds"],
        "effective_running_capacity_per_replica": capacity,
        "mean_load_cv_while_half_clients_active": float(np.mean(load_cv))
        if load_cv
        else None,
        "mean_active_adapter_versions_per_replica": float(np.mean(active_sets))
        if active_sets
        else None,
        "fraction_loaded_replica_observations_over_8_adapters": float(
            np.mean(np.array(active_sets) > 8)
        )
        if active_sets
        else None,
        "observed_peak_pool_active": max(
            (
                sum(s["active"] for s in o["replicas"])
                for o in data["observations"]
                if "replicas" in o
            ),
            default=0,
        ),
        "busy_snapshot_threshold": running_capacity,
        "reconstructed_busy_engine_seconds": int(engine_busy.sum()),
        "reconstructed_busy_engine_load_cv": float(
            np.mean(
                estimated_loads[:, engine_busy].std(axis=0)
                / estimated_loads[:, engine_busy].mean(axis=0)
            )
        )
        if engine_busy.any()
        else None,
        "reconstructed_busy_engine_mean_adapter_versions": float(
            estimated_sets[:, engine_busy].mean()
        )
        if engine_busy.any()
        else None,
        "reconstructed_busy_engine_fraction_over_8_adapters": float(
            np.mean(estimated_sets[:, engine_busy] > 8)
        )
        if engine_busy.any()
        else None,
        "busy_snapshot_count": len(busy),
        "mean_load_cv_while_pool_above_running_capacity": float(
            np.mean([loads.std() / loads.mean() for loads in busy_loads])
        )
        if busy_loads
        else None,
        "fraction_busy_snapshots_with_idle_replica": float(
            np.mean([np.any(loads == 0) for loads in busy_loads])
        )
        if busy_loads
        else None,
        "mean_busy_active_adapter_versions_per_replica": float(np.mean(busy_sets))
        if busy_sets
        else None,
        "fraction_busy_replica_observations_over_8_adapters": float(
            np.mean(np.array(busy_sets) > 8)
        )
        if busy_sets
        else None,
        "mean_replicas_per_adapter": float(
            np.mean([len(v) for v in placements.values()])
        ),
        "mean_replicas_per_prefix_group": float(
            np.mean([len(v) for v in prefix_placements.values()])
        ),
        "frontend_containers": len(set(data.get("frontends", []))),
        "prefix_lookup_count": lookup_count,
        "prefix_lookup_failures": sum(m["failures"] for m in lookups),
        "prefix_lookup_mean_s": sum(m["seconds"] for m in lookups) / lookup_count
        if lookup_count
        else None,
        "prefix_lookup_max_s": max((m["max_seconds"] for m in lookups), default=0),
        "rollout_mean_s": float(np.mean([r["seconds"] for r in data["rollouts"]]))
        if data.get("rollouts")
        else None,
        "rollout_p95_s": float(
            np.quantile([r["seconds"] for r in data["rollouts"]], 0.95)
        )
        if data.get("rollouts")
        else None,
        **counters,
    }


def series(data, width=20):
    rows = data["requests"]
    edges = np.arange(0, data["seconds"] + width, width)
    x, hits, latency, throughput = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if lo <= r["end"] < hi]
        x.append((lo + min(hi, data["seconds"])) / 2)
        hits.append(
            sum(r["cached_tokens"] for r in selected)
            / sum(r["prompt_tokens"] for r in selected)
            if selected
            else np.nan
        )
        latency.append(
            np.mean([r["seconds"] for r in selected]) if selected else np.nan
        )
        throughput.append(
            sum(r["decode_tokens"] for r in selected) / min(width, data["seconds"] - lo)
        )
    return np.array(x), np.array(hits), np.array(latency), np.array(throughput)


def save(fig, output, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plots(runs, summary, output):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
        }
    )
    for name, index, ylabel, field, percent in (
        ("kv-cache-hit-ratio", 1, "KV cache hit ratio", "kv_hit_ratio", True),
        (
            "request-completion-time",
            2,
            "Mean request completion time (s)",
            "latency_mean_s",
            False,
        ),
        (
            "output-throughput",
            3,
            "Completed output tokens / second",
            "tokens_per_second",
            False,
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for arm, data in runs.items():
            values = series(data)
            avg = summary[arm][field]
            label = (
                f"{LABELS[arm]} (overall {avg:.1%})"
                if percent
                else f"{LABELS[arm]} (overall {avg:,.1f})"
            )
            ax.plot(
                values[0],
                values[index],
                STYLES[arm],
                color=COLORS[arm],
                label=label,
                linewidth=2,
            )
        ax.set(xlabel="Time since workload start (seconds)", ylabel=ylabel)
        if percent:
            ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=9)
        save(fig, output, name)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for arm, data in runs.items():
        previous = totals(data["baseline"])
        xs, ys, evictions, cumulative_hits = [], [], [], []
        initial_hits = previous["request_lora_hits"]
        last_time = 0
        # Aggregate observations into >=10s intervals for legible admission rates.
        observations = [
            *data["observations"],
            {"elapsed": data["seconds"], "replicas": data["final"]},
        ]
        for index, observation in enumerate(observations):
            elapsed = observation["elapsed"] - last_time
            if "replicas" not in observation or elapsed <= 0:
                continue
            if elapsed < 10 and index != len(observations) - 1:
                continue
            current = totals(observation["replicas"])
            delta = {k: current[k] - previous[k] for k in COUNTERS}
            total = delta["request_lora_hits"] + delta["request_lora_misses"]
            gap = elapsed > 20
            if gap:
                xs.append(observation["elapsed"] - 1e-6)
                ys.append(np.nan)
                evictions.append(np.nan)
                cumulative_hits.append(np.nan)
            xs.append(observation["elapsed"])
            cumulative_hits.append(current["request_lora_hits"] - initial_hits)
            ys.append(
                delta["request_lora_hits"] / total if total and not gap else np.nan
            )
            evictions.append(delta["gpu_evictions"] / elapsed if not gap else np.nan)
            previous, last_time = current, observation["elapsed"]
        axes[0].plot(
            xs,
            ys,
            STYLES[arm],
            color=COLORS[arm],
            label=f"{LABELS[arm]} (overall {summary[arm]['lora_hit_ratio']:.1%})",
            linewidth=2,
        )
        axes[1].plot(
            xs,
            cumulative_hits,
            STYLES[arm],
            color=COLORS[arm],
            label=f"{LABELS[arm]} ({summary[arm]['request_lora_hits']:,} total)",
            linewidth=2,
        )
        axes[2].plot(
            xs,
            evictions,
            STYLES[arm],
            color=COLORS[arm],
            label=f"{LABELS[arm]} ({summary[arm]['gpu_evictions']:,} total)",
            linewidth=2,
        )
    axes[0].set(ylabel="Hot LoRA hit ratio\n(physical engine requests)", ylim=(0, 1))
    axes[1].set(ylabel="Cumulative hot LoRA request hits")
    axes[2].set(
        xlabel="Time since workload start (seconds)",
        ylabel="GPU adapter evictions / second",
    )
    for ax in axes:
        ax.legend(fontsize=9)
    save(fig, output, "lora-cache-and-evictions")

    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.3 * len(runs), 3.8), sharey=True, squeeze=False
    )
    for ax, (arm, data) in zip(axes[0], runs.items()):
        observations = [o for o in data["observations"] if "replicas" in o]
        hosts = sorted(s["replica"] for s in data["baseline"])
        for i, host in enumerate(hosts):
            times, active = [], []
            for o in observations:
                if times and o["elapsed"] - times[-1] > 15:
                    times.append(o["elapsed"] - 1e-6)
                    active.append(np.nan)
                times.append(o["elapsed"])
                active.append(
                    next(s["active"] for s in o["replicas"] if s["replica"] == host)
                )
            ax.plot(
                times,
                active,
                alpha=0.7,
                linewidth=1,
            )
        ax.set(
            title=LABELS[arm],
            xlabel="Time (s)",
            ylabel="Outstanding requests / replica",
        )
        ax.axhline(64, color="#777777", linestyle=":", linewidth=1)
    save(fig, output, "replica-load")

    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.3 * len(runs), 4), sharey=True, squeeze=False
    )
    for ax, (arm, data) in zip(axes[0], runs.items()):
        times, loads, _ = engine_intervals(data)
        for counts in loads:
            ax.plot(times, counts, alpha=0.7, linewidth=1)
        ax.set(
            title=LABELS[arm],
            xlabel="Time (s)",
            ylabel="Estimated requests in engine / replica",
        )
    fig.suptitle("Reconstructed successful-attempt intervals (1-second samples)")
    save(fig, output, "reconstructed-engine-load")

    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, data in runs.items():
        values = np.sort([r["seconds"] for r in data["requests"]])
        ax.plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            STYLES[arm],
            color=COLORS[arm],
            label=LABELS[arm],
            linewidth=2,
        )
    ax.set(
        xlabel="Request completion time (seconds)",
        ylabel="Fraction of requests completed",
        ylim=(0, 1),
    )
    ax.legend()
    save(fig, output, "latency-cdf")

    if all(data.get("rollouts") for data in runs.values()):
        fig, axes = plt.subplots(
            len(runs), 1, figsize=(11, 3.8 * len(runs)), squeeze=False, sharex=True
        )
        for ax, (arm, data) in zip(axes[:, 0], runs.items()):
            for row in data["rollouts"]:
                ax.broken_barh(
                    [(row["start"], row["end"] - row["start"])],
                    (row["client"] - 0.35, 0.7),
                    facecolors=COLORS[arm],
                    alpha=min(1.0, 0.45 + 0.2 * row["round"]),
                )
            ax.set(
                title=LABELS[arm], ylabel="Independent DAPO client", yticks=range(16)
            )
        axes[-1, 0].set_xlabel("Time since workload start (s)")
        save(fig, output, "client-rollout-timeline")

        fig, ax = plt.subplots(figsize=(10, 4))
        for arm, data in runs.items():
            observations = [o for o in data["observations"] if "replicas" in o]
            x = [o["elapsed"] for o in observations]
            y = [
                np.mean([len(r["adapters"]) for r in o["replicas"]])
                for o in observations
            ]
            # Missing intervals must not look like measured continuous data.
            for i in range(1, len(x)):
                if x[i] - x[i - 1] > 15:
                    y[i] = np.nan
            ax.plot(x, y, STYLES[arm], color=COLORS[arm], label=LABELS[arm])
        ax.axhline(8, color="black", linestyle=":", label="8 GPU adapter slots")
        ax.set(
            xlabel="Time since workload start (s)",
            ylabel="Mean active adapter versions per replica",
        )
        ax.legend()
        save(fig, output, "adapter-working-set")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--arms", nargs="+", choices=tuple(LABELS), default=["default", "lora", "kv"]
    )
    args = parser.parse_args()
    runs = {
        arm: json.loads((args.source / f"{arm}.json").read_text()) for arm in args.arms
    }
    assert len({d["trace_sha256"] for d in runs.values()}) == 1
    assert (
        len({tuple(sorted(s["replica"] for s in d["baseline"])) for d in runs.values()})
        == 1
    ), "Policies must run sequentially on the same eight replicas"
    summary = {arm: summarize(d) for arm, d in runs.items()}
    (args.source / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output = args.source / "plots"
    output.mkdir(exist_ok=True)
    plots(runs, summary, output)
    with (args.source / "requests.csv").open("w") as out:
        fields = [
            "arm",
            "id",
            "client",
            "turn",
            "adapter_index",
            "conversation",
            "version",
            "start",
            "end",
            "seconds",
            "prompt_tokens",
            "decode_tokens",
            "cached_tokens",
        ]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for arm, data in runs.items():
            writer.writerows(
                {k: arm if k == "arm" else r.get(k) for k in fields}
                for r in data["requests"]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
