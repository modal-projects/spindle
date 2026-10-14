"""Build the paper's auditable summaries and figures from completed raw arms."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from cache_report import summarize, totals


ARMS = (
    "kv",
    "kv-cohort",
    "stable-cohort",
    "work-cohort",
    "stable",
    "balanced",
    "work",
    "domains",
    "default",
)
LABELS = {
    "kv": "KV",
    "kv-cohort": "KV+P",
    "stable-cohort": "Domains+P",
    "work-cohort": "Work+P",
    "stable": "Domains",
    "balanced": "Unstable domains",
    "work": "Work",
    "domains": "Basic domains",
    "default": "Default",
}
COLORS = {
    "kv": "#bc3939",
    "kv-cohort": "#24845b",
    "stable-cohort": "#2459a6",
    "work-cohort": "#9a6523",
    "stable": "#663d9c",
    "balanced": "#784186",
    "work": "#a06a25",
    "domains": "#336ca6",
    "default": "#666666",
}
STYLES = {"kv": "--", "kv-cohort": "-.", "stable-cohort": "-", "work-cohort": ":"}


def analyzed(data):
    result = summarize(data)
    rows = data["requests"]
    versions = {}
    for row in rows:
        versions.setdefault(row["model_id"], set()).add(row["version"])
    result["distinct_adapter_versions"] = sum(len(v) for v in versions.values())
    result["distinct_model_ids"] = len(versions)
    result["clients_with_multiple_versions"] = sum(
        len(v) > 1 for v in versions.values()
    )
    auxiliary = sum(
        p["prompt_tokens"] - p["meta"]["cached_tokens"]
        for p in data.get("prefills", [])
    )
    result["auxiliary_uncached_prefill_tokens"] = auxiliary
    result["total_uncached_prefill_tokens"] = (
        result["prompt_tokens"] - result["cached_tokens"] + auxiliary
    )
    result["prefill_fraction_of_logical_input"] = (
        result["total_uncached_prefill_tokens"] / result["prompt_tokens"]
    )
    generation_span = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    result["generation_span_s"] = generation_span
    result["generation_span_tokens_per_second"] = (
        result["output_tokens"] / generation_span
    )
    result["lora_hit_count"] = result["request_lora_hits"]
    result["lora_miss_count"] = result["request_lora_misses"]
    result["independent_round_overlap"] = any(
        a["client"] != b["client"]
        and a["round"] > b["round"]
        and b["start"] < a["start"] < b["end"]
        for a in data.get("rollouts", [])
        for b in data.get("rollouts", [])
    )
    result["per_round"] = {}
    for turn in sorted({r["turn"] for r in rows}):
        subset = [r for r in rows if r["turn"] == turn]
        result["per_round"][turn] = {
            "requests": len(subset),
            "latency_mean_s": float(np.mean([r["seconds"] for r in subset])),
            "latency_p95_s": float(np.quantile([r["seconds"] for r in subset], 0.95)),
            "kv_hit_ratio": sum(r["cached_tokens"] for r in subset)
            / sum(r["prompt_tokens"] for r in subset),
        }
    result["per_dialogue_turn"] = {}
    for turn in sorted({r.get("dialogue_turn", 0) for r in rows}):
        subset = [r for r in rows if r.get("dialogue_turn", 0) == turn]
        result["per_dialogue_turn"][turn] = {
            "requests": len(subset),
            "latency_mean_s": float(np.mean([r["seconds"] for r in subset])),
            "kv_hit_ratio": sum(r["cached_tokens"] for r in subset)
            / sum(r["prompt_tokens"] for r in subset),
            "input_lengths": sorted({r["prompt_tokens"] for r in subset}),
        }
    result["per_client_rollout_mean_s"] = {
        i: float(np.mean([r["seconds"] for r in data["rollouts"] if r["client"] == i]))
        for i in sorted({r["client"] for r in data.get("rollouts", [])})
    }
    if data["config"].get("dialogue_turns", 1) > 1:
        rollouts = {(r["client"], r["round"]): r for r in data["rollouts"]}
        final_turn = max(r["dialogue_turn"] for r in rows)
        finished = [r for r in rows if r["dialogue_turn"] == final_turn]
        result["mean_trajectory_completion_s"] = float(
            np.mean(
                [r["end"] - rollouts[r["client"], r["turn"]]["start"] for r in finished]
            )
        )
        result["mean_finished_trajectory_wait_for_rollout_s"] = float(
            np.mean(
                [rollouts[r["client"], r["turn"]]["end"] - r["end"] for r in finished]
            )
        )
    snapshots = [
        s for o in data["observations"] if "replicas" in o for s in o["replicas"]
    ]
    sizes = [
        len(
            json.dumps(
                {k: v for k, v in s.items() if k not in {"gpu_lora", "engine"}},
                separators=(",", ":"),
            ).encode()
        )
        for s in snapshots
    ]
    result["routing_state_json_mean_bytes"] = float(np.mean(sizes)) if sizes else None
    native = [
        [
            sum(
                load["num_running_reqs"] + load["num_waiting_reqs"]
                for load in state["engine"]["loads"]
            )
            for state in observation["replicas"]
        ]
        for observation in data["observations"]
        if "replicas" in observation
    ]
    busy_native = [
        np.array(v) for v in native if sum(v) >= result["busy_snapshot_threshold"]
    ]
    result["native_busy_snapshot_count"] = len(busy_native)
    result["native_busy_load_cv"] = (
        float(np.mean([v.std() / v.mean() for v in busy_native]))
        if busy_native
        else None
    )
    result["native_busy_idle_fraction"] = (
        float(np.mean([np.any(v == 0) for v in busy_native])) if busy_native else None
    )
    queued = [
        load
        for state in snapshots
        for load in state["engine"]["loads"]
        if load["num_waiting_reqs"] > 0
    ]
    result["native_queued_replica_observations"] = len(queued)
    result["native_mean_running_requests_when_queued"] = (
        float(np.mean([v["num_running_reqs"] for v in queued])) if queued else None
    )
    result["native_mean_waiting_requests_when_queued"] = (
        float(np.mean([v["num_waiting_reqs"] for v in queued])) if queued else None
    )
    result["native_fraction_queued_below_half_running_capacity"] = (
        float(
            np.mean(
                [v["num_running_reqs"] < v["max_running_requests"] / 2 for v in queued]
            )
        )
        if queued
        else None
    )
    return result


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def paired_comparisons(summaries):
    pairs = {}
    metrics = (
        "tokens_per_second",
        "latency_mean_s",
        "latency_p95_s",
        "rollout_mean_s",
        "gpu_loads",
        "gpu_evictions",
        "load_host_seconds",
        "kv_hit_ratio",
        "lora_hit_ratio",
        "total_uncached_prefill_tokens",
    )
    for folder, arms in summaries.items():
        baseline, candidate = (
            ("kv", "stable")
            if folder.startswith("serving64-final")
            else ("kv-cohort", "stable-cohort")
        )
        if baseline not in arms or candidate not in arms:
            continue
        before, after = arms[baseline], arms[candidate]
        row = {
            "baseline": baseline,
            "candidate": candidate,
            "percent_change": {
                metric: 100 * (after[metric] / before[metric] - 1)
                for metric in metrics
                if before[metric] not in (None, 0)
            },
            "kv_hit_percentage_point_change": 100
            * (after["kv_hit_ratio"] - before["kv_hit_ratio"]),
        }
        clients = before["per_client_rollout_mean_s"]
        if clients:
            speedups = {
                str(i): value / after["per_client_rollout_mean_s"][i]
                for i, value in clients.items()
            }
            row.update(
                per_client_rollout_speedup=speedups,
                clients_improved=sum(v > 1 for v in speedups.values()),
                min_client_speedup=min(speedups.values()),
                max_client_speedup=max(speedups.values()),
            )
        pairs[folder] = row
    return pairs


def latex_tables(summaries, output):
    """Keep every completed trial visible rather than pooling request samples."""
    names = {
        "warm": "Warmed reuse",
        "turnover": "Version turnover",
        "dialogue": "Dialogue",
    }
    primary = [
        (f"{w}-h2-r{r}", f"{label} {r + 1}")
        for w, label in names.items()
        for r in (0, 1)
    ] + [(f"serving64-final-r{r}", f"64 versions {r + 1}") for r in (0, 1)]
    primary += [("long-decode-r0", "Long 512"), ("long-decode1024-r0", "Long 1,024")]
    header = [
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{llrrrrrrr}\toprule",
        r"Workload / trial & Policy & Replay (s) & Output tok/s & Mean (s) & p95 (s) & Rollout (s) & KV hit & Retries\\\midrule",
    ]
    rows = []
    for folder, label in primary:
        for arm, s in summaries.get(folder, {}).items():
            rollout = s.get("rollout_mean_s")
            rollout = f"{rollout:.1f}" if rollout is not None else "--"
            rows.append(
                f"{label} & {LABELS[arm]} & {s['seconds']:.1f} & "
                f"{s['tokens_per_second']:.0f} & {s['latency_mean_s']:.1f} & "
                f"{s['latency_p95_s']:.1f} & {rollout} & {s['kv_hit_ratio']:.3f} & "
                f"{s['reroutes']} " + r"\\"
            )
    footer = [
        r"\bottomrule\end{tabular}",
        r"\caption{Completed HTTP/2 comparisons, reported separately by trial. All times include any cohort preparation. Replay throughput includes recorded update waits for RL workloads; rollout latency excludes those waits. Trial 2 reverses policy order. No confidence interval is inferred from individual requests.}\label{tab:primary}",
        r"\end{table*}",
    ]
    (output / "primary-table.tex").write_text("\n".join(header + rows + footer) + "\n")
    header = [
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{llrrrrrrr}\toprule",
        r"Workload / trial & Policy & Hot hits & Misses & Loads & Evicts & Register & Seeds & Uncached input\\\midrule",
    ]
    rows = []
    for folder, label in [("turnover-r0", "Turnover HTTP/1")] + primary:
        for arm, s in summaries.get(folder, {}).items():
            rows.append(
                f"{label} & {LABELS[arm]} & {s['request_lora_hits']} & "
                f"{s['request_lora_misses']} & {s['gpu_loads']} & {s['gpu_evictions']} & "
                f"{s['registration_calls']} & {s['cohort_prefills']} & "
                f"{100 * s['prefill_fraction_of_logical_input']:.1f}\\% " + r"\\"
            )
    footer = [
        r"\bottomrule\end{tabular}",
        r"\caption{State movement and cache accounting. Hot hits and misses count engine admissions, including auxiliary seeds and retries. Uncached input includes auxiliary seed computation and is divided by logical requested input tokens. Register counts first-use sidecar registration calls, not GPU loads.}\label{tab:state}",
        r"\end{table*}",
    ]
    (output / "state-table.tex").write_text("\n".join(header + rows + footer) + "\n")

    rows = []
    for folder, length in (("long-decode-r0", "512"), ("long-decode1024-r0", "1,024")):
        for arm, s in summaries.get(folder, {}).items():
            rows.append(
                f"{length} & {LABELS[arm]} & {s['requests']} & "
                f"{s['steady_completed_requests']} & "
                f"{s['steady_output_tokens_per_second']:.0f} & "
                f"{s['steady_latency_mean_s']:.1f} & "
                f"{1000 * s['gpu_loads'] / s['requests']:.1f} " + r"\\"
            )
    text = (
        [
            r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llrrrrr}\toprule",
            r"Output tokens & Policy & Requests & Window completions & Window tok/s & Window mean (s) & Loads / 1k\\\midrule",
        ]
        + rows
        + [
            r"\bottomrule\end{tabular}",
            r"\caption{Longer-output controls. Window metrics use completions from 60 seconds to the arrival-window end: 300 seconds for 512-token outputs and 180 seconds for 1,024-token outputs. Admitted work drains afterward. Total work depends on policy, so loads are normalized by logical requests. Each output length has one trial.}\label{tab:long}",
            r"\end{table*}",
        ]
        if rows
        else []
    )
    (output / "long-table.tex").write_text("\n".join(text) + "\n")


def rolling(data, width=20, max_observation_gap=None):
    rows = data["requests"]
    x = np.arange(5, data["seconds"] + 1, 5)
    metrics = {k: [] for k in ("cache", "latency", "throughput", "lora")}
    polls = (
        [(0, totals(data["baseline"]))]
        + [
            (o["elapsed"], totals(o["replicas"]))
            for o in data["observations"]
            if "replicas" in o
        ]
        + [(data["seconds"], totals(data["final"]))]
    )
    times = np.array([t for t, _ in polls])
    hits = np.array([s["request_lora_hits"] for _, s in polls])
    admits = hits + np.array([s["request_lora_misses"] for _, s in polls])
    gaps = [
        (a, b)
        for a, b in zip(times, times[1:])
        if max_observation_gap is not None and b - a > max_observation_gap
    ]
    for end in x:
        begin = max(0, end - width)
        selected = [r for r in rows if begin <= r["end"] < end]
        metrics["cache"].append(
            sum(r["cached_tokens"] for r in selected)
            / sum(r["prompt_tokens"] for r in selected)
            if selected
            else np.nan
        )
        metrics["latency"].append(
            np.mean([r["seconds"] for r in selected]) if selected else np.nan
        )
        metrics["throughput"].append(
            sum(r["decode_tokens"] for r in selected) / (end - begin)
        )
        count = np.interp(end, times, admits) - np.interp(begin, times, admits)
        metrics["lora"].append(
            (np.interp(end, times, hits) - np.interp(begin, times, hits)) / count
            if count > 0 and not any(a < end and b > begin for a, b in gaps)
            else np.nan
        )
    return x, metrics


def export_plot_values(runs, output, max_observation_gap=None):
    """Share plotted values without prompts, generated text, or checkpoint paths."""
    values = {}
    for workload, arms in runs.items():
        for arm, data in arms.items():
            times, curves = rolling(data, max_observation_gap=max_observation_gap)
            values.setdefault(workload, {})[arm] = {
                "time_seconds": times.tolist(),
                "window_seconds": 20,
                "max_observation_gap_seconds": max_observation_gap,
                "rolling": {
                    metric: [float(v) if np.isfinite(v) else None for v in curve]
                    for metric, curve in curves.items()
                },
                "sorted_request_latency_seconds": sorted(
                    r["seconds"] for r in data["requests"]
                ),
            }
    payload = json.dumps(values, separators=(",", ":"), allow_nan=False).encode()
    (output / "plot-values.json.gz").write_bytes(gzip.compress(payload, mtime=0))


def figures(runs, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    for x, width, title, body in (
        (
            0.05,
            2.6,
            "Independent frontends",
            "One router per pool\nLocal observations + reservations\nPlan chunks; prepare prefixes",
        ),
        (
            3.55,
            2.2,
            "Modal Flash",
            "Authenticated request path\nExact destination header\nExisting proxy",
        ),
        (
            6.65,
            3.25,
            "Inference replicas",
            "Sidecar: version registration\nSGLang: batching + GPU LoRAs\nEngine-owned cache observations",
        ),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.85),
                width,
                1.65,
                boxstyle="round,pad=0.04",
                facecolor="#f0f3f7",
                edgecolor="#718096",
                lw=0.8,
            )
        )
        ax.text(
            x + width / 2,
            2.21,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=8,
        )
        ax.text(
            x + width / 2,
            1.5,
            body,
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.5,
        )
    for a, b in ((2.7, 3.48), (5.8, 6.57)):
        ax.add_patch(
            FancyArrowPatch(
                (a, 1.65),
                (b, 1.65),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.1,
                color="#2459a6",
            )
        )
    ax.plot([8.25, 8.25, 1.3], [0.78, 0.4, 0.4], color="#24845b", ls="--", lw=1)
    ax.annotate(
        "",
        xy=(1.3, 0.78),
        xytext=(1.3, 0.4),
        arrowprops=dict(arrowstyle="->", color="#24845b", ls="--", lw=1),
    )
    ax.text(
        4.8,
        0.02,
        "Background observations through Flash; no cluster-wide request scheduler",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#246648",
    )
    fig.tight_layout(pad=0.1)
    save(fig, output / "architecture")
    workloads = [w for w in ("warm", "turnover", "dialogue") if w + "-h2-r0" in runs]
    if workloads:
        fig, axes = plt.subplots(
            4, len(workloads), figsize=(7.1, 6.5), squeeze=False, sharex="col"
        )
        for col, workload in enumerate(workloads):
            for arm in ("kv-cohort", "stable-cohort"):
                if arm not in runs[workload + "-h2-r0"]:
                    continue
                data = runs[workload + "-h2-r0"][arm]
                x, values = rolling(data)
                for row, key in enumerate(("cache", "latency", "lora", "throughput")):
                    axes[row, col].plot(
                        x,
                        values[key],
                        color=COLORS[arm],
                        label=LABELS[arm],
                        lw=1.2,
                        ls=STYLES.get(arm, "-"),
                    )
                    axes[row, col].grid(alpha=0.18)
                    axes[row, col].tick_params(labelsize=7)
            axes[0, col].set_title(
                {
                    "warm": "Warmed reuse",
                    "turnover": "Skew + version turnover",
                    "dialogue": "Three-turn dialogue",
                }[workload],
                fontsize=9,
            )
            axes[-1, col].set_xlabel("Time (s)")
            axes[0, col].set_ylim(0, 1.03)
            axes[2, col].set_ylim(0, 1.03)
        for ax, label in zip(
            axes[:, 0],
            (
                "KV token hit ratio",
                "Mean completion\nlatency (s)",
                "Hot-LoRA admission\nratio",
                "Useful output\ntokens/s",
            ),
        ):
            ax.set_ylabel(label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.tight_layout(h_pad=0.6, w_pad=0.7)
        save(fig, output / "time-series")

    if "turnover-r0" in runs:
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
        for arm, data in runs["turnover-r0"].items():
            polls = [(0, data["baseline"])] + [
                (o["elapsed"], o["replicas"])
                for o in data["observations"]
                if "replicas" in o
            ]
            before = totals(data["baseline"])["gpu_loads"]
            axes[0].plot(
                [t for t, _ in polls],
                [totals(s)["gpu_loads"] - before for _, s in polls],
                color=COLORS[arm],
                label=LABELS[arm],
                ls=STYLES.get(arm, "-"),
            )
            load = [np.array([s["active"] for s in states]) for _, states in polls]
            axes[1].plot(
                [t for t, _ in polls],
                [a.std() / a.mean() if a.sum() >= 288 else np.nan for a in load],
                color=COLORS[arm],
                ls=STYLES.get(arm, "-"),
            )
            values = [
                np.mean([r["seconds"] for r in data["rollouts"] if r["client"] == i])
                for i in range(16)
            ]
            axes[2].plot(
                range(16),
                values,
                marker=".",
                ms=3,
                lw=1,
                color=COLORS[arm],
                ls=STYLES.get(arm, "-"),
            )
        axes[0].set(ylabel="Cumulative GPU adapter loads", xlabel="Time (s)")
        axes[1].set(ylabel="Replica load CV (busy pool)", xlabel="Time (s)")
        axes[2].set(ylabel="Mean client rollout (s)", xlabel="Client")
        for ax in axes:
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 1.11),
        )
        fig.tight_layout(w_pad=0.8)
        save(fig, output / "turnover-mechanisms")

    if workloads:
        fig, axes = plt.subplots(1, len(workloads), figsize=(7.1, 2.35), squeeze=False)
        axes = axes[0]
        for col, workload in enumerate(workloads):
            for index, arm in enumerate(("kv-cohort", "stable-cohort")):
                samples = []
                for trial in (0, 1):
                    data = runs.get(workload + f"-h2-r{trial}", {}).get(arm)
                    if data:
                        samples.append(
                            np.mean([r["seconds"] for r in data["rollouts"]])
                        )
                if samples:
                    axes[col].bar(
                        index,
                        np.mean(samples),
                        color=COLORS[arm],
                        alpha=0.65,
                        width=0.65,
                    )
                    axes[col].scatter(
                        index + np.linspace(-0.09, 0.09, len(samples)),
                        samples,
                        color="black",
                        s=12,
                        zorder=5,
                    )
            axes[col].set_xticks(range(2), ["KV+P", "Domains+P"], rotation=12)
            axes[col].set_title(workload.capitalize())
            axes[col].grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Mean rollout completion (s)")
        fig.tight_layout(w_pad=1)
        save(fig, output / "rollout-comparison")

        fig, axes = plt.subplots(1, len(workloads), figsize=(7.1, 2.25), squeeze=False)
        for col, workload in enumerate(workloads):
            for arm in ("kv-cohort", "stable-cohort"):
                data = runs[workload + "-h2-r0"].get(arm)
                if not data:
                    continue
                values = np.sort([r["seconds"] for r in data["requests"]])
                axes[0, col].plot(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    color=COLORS[arm],
                    ls=STYLES[arm],
                    label=LABELS[arm],
                )
            axes[0, col].set(title=workload.capitalize(), xlabel="Request latency (s)")
            axes[0, col].grid(alpha=0.2)
        axes[0, 0].set_ylabel("Empirical CDF")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.tight_layout(w_pad=1)
        save(fig, output / "request-cdf")

    if "serving64-final-r0" in runs:
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
        for index, arm in enumerate(("default", "kv", "stable")):
            samples = [
                analyzed(runs[f"serving64-final-r{trial}"][arm])
                for trial in (0, 1)
                if arm in runs.get(f"serving64-final-r{trial}", {})
            ]
            for ax, field, label in zip(
                axes,
                ("tokens_per_second", "latency_p95_s", "gpu_loads"),
                ("Useful output tokens/s", "p95 latency (s)", "GPU adapter loads"),
            ):
                if samples:
                    values = [s[field] for s in samples]
                    ax.bar(index, np.mean(values), color=COLORS[arm], alpha=0.65)
                    ax.scatter(
                        index + np.linspace(-0.08, 0.08, len(values)),
                        values,
                        color="black",
                        s=12,
                        zorder=5,
                    )
                ax.set(ylabel=label)
                ax.set_xticks(range(3), ["Default", "KV", "Domains"], rotation=12)
                ax.grid(axis="y", alpha=0.2)
        fig.tight_layout(w_pad=1)
        save(fig, output / "general-serving")

        fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0), sharex=True)
        for arm in ("default", "kv", "stable"):
            data = runs["serving64-final-r0"].get(arm)
            if not data:
                continue
            x, values = rolling(data)
            for ax, key, label in zip(
                axes.flat,
                ("cache", "latency", "lora", "throughput"),
                (
                    "KV token hit ratio",
                    "Mean completion latency (s)",
                    "Hot-LoRA admission ratio",
                    "Useful output tokens/s",
                ),
            ):
                ax.plot(
                    x,
                    values[key],
                    color=COLORS[arm],
                    label=LABELS[arm],
                    ls={"default": ":", "kv": "--", "stable": "-"}[arm],
                )
                ax.set_ylabel(label)
                ax.grid(alpha=0.2)
        for ax in axes[1]:
            ax.set_xlabel("Time (s)")
        axes[0, 0].set_ylim(0, 1.03)
        axes[1, 0].set_ylim(0, 1.03)
        axes[0, 1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        save(fig, output / "general-time-series")

    if "dialogue-h2-r0" in runs:
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0))
        for arm in ("kv-cohort", "stable-cohort"):
            data = runs["dialogue-h2-r0"].get(arm)
            if not data:
                continue
            by_turn = analyzed(data)["per_dialogue_turn"]
            for ax, key, label in zip(
                axes,
                ("kv_hit_ratio", "latency_mean_s"),
                ("KV token hit ratio", "Mean latency (s)"),
            ):
                ax.plot(
                    [int(t) + 1 for t in by_turn],
                    [s[key] for s in by_turn.values()],
                    color=COLORS[arm],
                    label=LABELS[arm],
                    ls=STYLES[arm],
                    marker=".",
                )
                ax.set(xlabel="Dialogue turn", ylabel=label, xticks=[1, 2, 3])
                ax.grid(alpha=0.2)
        axes[0].set_ylim(0, 1.03)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.08),
            fontsize=7,
        )
        fig.tight_layout(w_pad=0.4)
        save(fig, output / "dialogue-turns")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, summaries = {}, {}
    folders = (
        "pilot",
        "cohort-pilot",
        "serving64-r0",
        "serving64-balanced-pilot",
        "serving64-single-h2-client-pilot",
        "turnover-r0",
        "warm-r0",
        "dialogue-r0",
        "turnover-r1",
        "warm-r1",
        "dialogue-r1",
        "serving64-final-r0",
        "serving64-final-r1",
        "warm-h2-r0",
        "warm-h2-r1",
        "turnover-h2-r0",
        "turnover-h2-r1",
        "dialogue-h2-r0",
        "dialogue-h2-r1",
        "long-decode-r0",
        "long-decode1024-r0",
    )
    for folder in folders:
        for arm in ARMS:
            path = args.root / folder / (arm + ".json")
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            result = analyzed(data)
            result.update(
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                workload_sha256=data.get("workload_sha256"),
                source_sha256=data["config"].get("source_sha256"),
                config=data["config"],
                raw_path=str(path),
            )
            summaries.setdefault(folder, {})[arm] = result
            runs.setdefault(folder, {})[arm] = data
        hashes = {d.get("workload_sha256") for d in runs.get(folder, {}).values()}
        assert len(hashes) <= 1, (folder, "Workload definitions differ", hashes)
    for stem in ("warm-h2", "turnover-h2", "dialogue-h2", "serving64-final"):
        hashes = {
            data["workload_sha256"]
            for trial in (0, 1)
            for data in runs.get(f"{stem}-r{trial}", {}).values()
        }
        assert len(hashes) <= 1, (stem, "Repeated workload definitions differ", hashes)
    (args.output / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "paired-comparisons.json").write_text(
        json.dumps(paired_comparisons(summaries), indent=2) + "\n"
    )
    flat = [
        dict(
            workload=w,
            policy=a,
            **{k: v for k, v in s.items() if not isinstance(v, (dict, list))},
        )
        for w, arms in summaries.items()
        for a, s in arms.items()
    ]
    columns = list(dict.fromkeys(k for row in flat for k in row))
    with (args.output / "metrics.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    figures(runs, args.output)
    export_plot_values(runs, args.output)
    latex_tables(summaries, args.output)
    for workload, arms in summaries.items():
        print(workload)
        for arm, s in arms.items():
            print(
                f"  {arm:15s} {s['requests']:5d} req {s['seconds']:7.1f}s {s['tokens_per_second']:7.1f} tok/s mean {s['latency_mean_s']:6.1f}s p95 {s['latency_p95_s']:6.1f}s KV {s['kv_hit_ratio']:.3f} hot {s['lora_hit_ratio']:.3f} loads {s['gpu_loads']:4d} retries {s['reroutes']:3d}"
            )


if __name__ == "__main__":
    main()
