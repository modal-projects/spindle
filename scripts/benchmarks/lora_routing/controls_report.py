"""Validate all saved follow-up arms, including the simple controls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from paper_results import analyzed, export_plot_values
from supervise import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, runs, failures = {}, {}, {}
    for path in sorted(args.root.glob("*/*.json")):
        if path.parent.name in {"calibration", "report"}:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "requests" not in data or "arm" not in data:
            continue
        if data.get("error"):
            failures[str(path)] = data["error"]
            continue
        result = analyzed(data)
        attempts = [a["attrs"] for row in data["requests"] for a in row["attempts"]]
        result["explicit_destination_attempts"] = sum(
            a.get("selected_replica") is not None for a in attempts
        )
        result["native_destination_attempts"] = (
            len(attempts) - result["explicit_destination_attempts"]
        )
        result["admission_rejections"] = sum(
            a.get("admission_rejected", False) for a in attempts
        )
        result["admission_checked_requests"] = sum(
            a.get("ok", False) and a.get("admission_checked", False) for a in attempts
        )
        result["admission_queue_fallback_requests"] = sum(
            a.get("ok", False) and a.get("routing", {}).get("admission_fallback", False)
            for a in attempts
        )
        result["transport_failures"] = sum("error_type" in a for a in attempts)
        result.update(
            raw_path=str(path),
            raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            config=data["config"],
            workload_sha256=data["workload_sha256"],
            source_sha256=data["config"]["source_sha256"],
        )
        summaries.setdefault(path.parent.name, {})[data["arm"]] = result
        runs.setdefault(path.parent.name, {})[data["arm"]] = data
    for name, arms in summaries.items():
        assert len({r["workload_sha256"] for r in arms.values()}) == 1, name
        print(name)
        for arm, r in arms.items():
            print(
                f"  {arm:22s} n={r['requests']:5d} t={r['seconds']:7.1f}s "
                f"tok/s={r['tokens_per_second']:7.1f} mean={r['latency_mean_s']:6.1f}s "
                f"p95={r['latency_p95_s']:6.1f}s loads={r['gpu_loads']:4d} "
                f"kv={r['kv_hit_ratio']:.3f} extra_http={r['physical_attempts'] - r['requests']}"
                f" admission_rejections={r['admission_rejections']}"
                f" native={r['native_destination_attempts']}"
            )
    write(args.output / "metrics.json", summaries)
    write(args.output / "failures.json", failures)
    flat = [
        dict(
            workload=w,
            policy=a,
            **{k: v for k, v in r.items() if not isinstance(v, (dict, list))},
        )
        for w, arms in summaries.items()
        for a, r in arms.items()
    ]
    with (args.output / "metrics.csv").open("w") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=list(dict.fromkeys(k for r in flat for k in r)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flat)
    export_plot_values(runs, args.output)


if __name__ == "__main__":
    main()
