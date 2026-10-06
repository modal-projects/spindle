"""Correlate client chunk hashes with backend arrivals; refuse mismatched data."""

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import statistics


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            result.append(json.loads(line[start:]))
        except ValueError:
            continue
    return result


def distribution(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[min(len(values) - 1, int(0.95 * len(values)))],
        "min": min(values),
        "max": max(values),
    }


def summarize(clients, backend, warmup):
    traces = [rows(p) for p in sorted(clients.glob("client-*.jsonl"))]
    assert len(traces) == 8 and all(
        any(r["event"] == "complete" for r in trace) for trace in traces
    ), "Run is incomplete"
    received = defaultdict(deque)
    for row in rows(backend):
        if row.get("mark") == "benchmark.backend_payload":
            received[(row["model_id"], row["sha256"])].append(row["arrived"])
    expected = Counter(
        (r["model_id"], h)
        for trace in traces
        for r in trace
        if r["event"] == "prepared"
        for h in r["sha256"]
    )
    actual = Counter({key: len(values) for key, values in received.items()})
    assert expected == actual, {
        "missing": dict(expected - actual),
        "extra": dict(actual - expected),
    }
    requests = []
    for trace in traces:
        prepared = {r["step"]: r for r in trace if r["event"] == "prepared"}
        starts = {r["step"]: r["time"] for r in trace if r["event"] == "forward_start"}
        ends = {r["step"]: r["time"] for r in trace if r["event"] == "forward_done"}
        train = {r["step"]: r["time"] for r in trace if r["event"] == "train_done"}
        for step, r in prepared.items():
            arrivals = [received[(r["model_id"], h)].popleft() for h in r["sha256"]]
            requests.append(
                {
                    "client": r["client"],
                    "step": step,
                    "tokens": r["input_tokens"],
                    "chunks": r["chunks"],
                    "first_backend_s": min(arrivals) - starts[step],
                    "last_backend_s": max(arrivals) - starts[step],
                    "forward_result_s": ends[step] - starts[step],
                    "train_result_s": train[step] - starts[step],
                }
            )
    measured = [r for r in requests if r["step"] >= warmup]
    return {
        "correctness": {
            "matched_chunks": sum(expected.values()),
            "matched_updates": len(requests),
            "input_tokens": sum(r["tokens"] for r in requests),
            "all_hashes_match": True,
        },
        "warmup_updates_per_client": warmup,
        "latency": {
            k: distribution([r[k] for r in measured])
            for k in [
                "first_backend_s",
                "last_backend_s",
                "forward_result_s",
                "train_result_s",
            ]
        },
        "requests": requests,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=Path, required=True)
    p.add_argument("--backend", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.clients, a.backend, a.warmup)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
