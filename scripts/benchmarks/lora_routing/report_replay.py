"""Compare matched inference replay waves, including every request group."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(runs):
    assert len(runs) == 2 and {r["mode"] for r in runs} == {"naive", "affinity"}
    assert len({r["workload_sha256"] for r in runs}) == 1
    assert len({r["max_tokens"] for r in runs}) == 1
    signatures = []
    summary, rows = {}, []
    for run in runs:
        groups = [g for w in run["waves"] for g in w["groups"]]
        signature = []
        for wave in run["waves"]:
            assert len(wave["groups"]) == 128
            assert {(g["client"], g["group"]) for g in wave["groups"]} == {
                (client, group) for client in range(16) for group in range(8)
            }
            assert wave["tokens"] == sum(g["tokens"] for g in wave["groups"])
            for group in wave["groups"]:
                assert 0 < group["tokens"] <= 8 * run["max_tokens"]
                assert math.isfinite(group["seconds"]) and group["seconds"] > 0
                signature.append(
                    (wave["wave"], group["client"], group["group"], group["version"])
                )
                rows.append(
                    {
                        "mode": run["mode"],
                        "wave": wave["wave"],
                        **{
                            k: group[k]
                            for k in ("client", "group", "version", "seconds", "tokens")
                        },
                    }
                )
        signatures.append(sorted(signature))
        latencies = sorted(g["seconds"] for g in groups)
        tokens = sum(w["tokens"] for w in run["waves"])
        seconds = sum(w["seconds"] for w in run["waves"])
        summary[run["mode"]] = {
            "output_tokens": tokens,
            "measured_seconds": seconds,
            "output_tokens_per_second": tokens / seconds,
            "group_mean_seconds": statistics.mean(latencies),
            "group_p95_seconds": latencies[math.ceil(len(latencies) * 0.95) - 1],
            "workload_sha256": run["workload_sha256"],
            "waves": [
                {k: v for k, v in w.items() if k != "groups"} for w in run["waves"]
            ],
        }
    assert signatures[0] == signatures[1]
    summary["affinity_vs_naive_ratios"] = {
        key: summary["affinity"][key] / summary["naive"][key]
        for key in (
            "output_tokens_per_second",
            "group_mean_seconds",
            "group_p95_seconds",
            "measured_seconds",
        )
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--router-commit", help="Revision used by the measured workers")
    args = parser.parse_args()
    raw = args.replay.read_bytes()
    summary, rows = summarize(json.loads(raw))
    summary["raw_replay_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.router_commit:
        summary["router_commit"] = args.router_commit
    if args.groups_csv:
        with args.groups_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
