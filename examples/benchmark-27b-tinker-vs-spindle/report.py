"""Report completed-group throughput and historical price estimates, without plots."""

import argparse
import gzip
import json
from pathlib import Path

# Rates used on 2026-10-08, not a claim about current prices. Edit for a new run.
GPU_SECOND_USD = 0.001261
TINKER_RATES = {
    "input": 1.86,
    "cached_input": 0.372,
    "output": 5.595,
    "training": 4.103,
}


def load_events(path):
    # A live writer may be partway through the final line.
    lines = path.read_text().splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return events


def summarize_tinker(path, first_update=1, last_update=None):
    events = load_events(path)
    completed = [
        e
        for e in events
        if e["event"] == "update_done"
        and e["update"] >= first_update
        and (last_update is None or e["update"] <= last_update)
    ]
    if not completed:
        raise ValueError("No completed Tinker updates in the requested window")
    last = max(e["update"] for e in completed)
    expected = set(range(first_update, last + 1))
    rollouts = [
        e for e in events if e["event"] == "rollout_done" and e["update"] in expected
    ]
    if {e["update"] for e in completed} != expected or {
        e["update"] for e in rollouts
    } != expected:
        raise ValueError(
            "Tinker window must contain consecutive completed updates and rollouts"
        )
    start = next(
        e["time"]
        for e in events
        if e["event"] == "rollout_start" and e["update"] == first_update
    )
    end = max(e["time"] for e in completed)
    output = sum(e["output_tokens"] for e in rollouts)
    inputs = sum(e["sampling_input_tokens"] for e in rollouts)
    training = sum(e["training_input_tokens"] for e in rollouts)
    low = (
        output * TINKER_RATES["output"]
        + inputs * TINKER_RATES["cached_input"]
        + training * TINKER_RATES["training"]
    ) / 1e6
    high = low + inputs * (TINKER_RATES["input"] - TINKER_RATES["cached_input"]) / 1e6
    return dict(
        updates=[first_update, last],
        start=start,
        end=end,
        seconds=end - start,
        output_tokens=output,
        sampling_input_tokens=inputs,
        training_tokens=training,
        aggregate_output_tps=output / (end - start),
        rollout_only_tps=output / sum(e["seconds"] for e in rollouts),
        estimated_cost_usd=[low, high],
        cost_per_million_output_tokens=[low / output * 1e6, high / output * 1e6],
    )


def summarize_spindle(path, cutoff=None, expected_clients=8):
    directories = sorted(path.glob("client-??"))
    if len(directories) != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} Spindle clients, found {len(directories)}"
        )
    traces = [load_events(d / "events.jsonl") for d in directories]
    if any(not any(e["event"] == "rollout_start" for e in trace) for trace in traces):
        raise ValueError("Wait until every client has started its first rollout")
    start = max(
        next(e["time"] for e in trace if e["event"] == "rollout_start")
        for trace in traces
    )
    finishes = [
        e["time"] for trace in traces for e in trace if e["event"] == "completed"
    ]
    if cutoff is None:
        if len(finishes) != expected_clients:
            raise ValueError(
                "Incomplete run: supply --cutoff to explicitly request an interim report"
            )
        cutoff = min(finishes)
    if cutoff <= start:
        raise ValueError("The cutoff must follow every client's first rollout")
    if finishes and cutoff > min(finishes):
        raise ValueError(
            "The cutoff must not extend beyond the first client's completion"
        )
    client_tokens = {}
    for directory in directories:
        tokens = 0
        for file in directory.glob("group-*.json.gz"):
            try:
                with gzip.open(file, "rt") as stream:
                    group = json.load(stream)
            except EOFError:
                # The live client has not finished writing this group yet.
                continue
            if start <= group["end"] <= cutoff:
                tokens += sum(len(s["tokens"]) for s in group["sequences"])
        client_tokens[directory.name] = tokens
    total = sum(client_tokens.values())
    if not total:
        raise ValueError("No completed group tokens in the requested window")
    lifetimes = json.loads((path / "gpu-lifetimes.json").read_text())
    gpu = [r for r in lifetimes if r["gpu_count"] and r["started_at"]]
    if not gpu or any(r["gpu_type"] != "H200" for r in gpu):
        raise ValueError(
            "Expected recorded H200 lifetimes; check collection errors and GPU pricing"
        )
    gpu_seconds = sum(
        r["gpu_count"]
        * max(0, min(r["finished_at"] or cutoff, cutoff) - max(r["started_at"], start))
        for r in gpu
    )
    if gpu_seconds <= 0:
        raise ValueError("No recorded GPU time overlaps the measurement window")
    seconds = cutoff - start
    return dict(
        start=start,
        end=cutoff,
        seconds=seconds,
        output_tokens=total,
        aggregate_output_tps=total / seconds,
        mean_output_tps_per_client=total / seconds / len(directories),
        per_client_output_tps={k: v / seconds for k, v in client_tokens.items()},
        gpu_seconds=gpu_seconds,
        gpu_cost_usd=gpu_seconds * GPU_SECOND_USD,
        gpu_cost_per_million_output_tokens=gpu_seconds * GPU_SECOND_USD / total * 1e6,
        completed_updates=[
            sum(e["event"] == "update_done" and e["time"] <= cutoff for e in trace)
            for trace in traces
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spindle", type=Path, help="run_spindle.py output directory")
    parser.add_argument(
        "--tinker", type=Path, required=True, help="Tinker events.jsonl"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Spindle window end, Unix seconds; required for interim runs",
    )
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--tinker-first-update", type=int, default=1)
    parser.add_argument("--tinker-last-update", type=int)
    args = parser.parse_args()
    result = {
        "tinker": summarize_tinker(
            args.tinker, args.tinker_first_update, args.tinker_last_update
        ),
        "rates": {
            "h200_gpu_second": GPU_SECOND_USD,
            "tinker_per_million": TINKER_RATES,
        },
        "notes": [
            "Completed-group counts can include tokens generated before the window or not yet trained.",
            "Spindle starts when the last client starts sampling; startup, staggering, CPU, memory and storage costs excluded.",
            "Tinker covers the selected completed updates, including training/publication; input cache hits unknown.",
            "Prices are historical estimates, not invoices. Resource collection may miss short-lived containers.",
        ],
    }
    if args.spindle:
        result["spindle"] = summarize_spindle(args.spindle, args.cutoff, args.clients)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
