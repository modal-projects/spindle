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
            "group_p50_seconds": statistics.median(latencies),
            "group_p75_seconds": latencies[math.ceil(len(latencies) * 0.75) - 1],
            "group_p95_seconds": latencies[math.ceil(len(latencies) * 0.95) - 1],
            "group_p99_seconds": latencies[math.ceil(len(latencies) * 0.99) - 1],
            "group_max_seconds": max(latencies),
            "groups_over_75_seconds": sum(value > 75 for value in latencies),
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


def summarize_observations(run, observations):
    """Reconstruct registrations only when recent histories are complete.

    The observer joined during wave zero. Its counters cannot isolate that
    wave from warmup. Subsequent registrations and the new version are covered.
    These are CPU registration RPC spans, not GPU transfer measurements.
    """
    first_version, new_version = (w["versions"][0] for w in run["waves"][:2])
    assert [w["versions"] for w in run["waves"]] == [
        [first_version],
        [new_version],
        [first_version],
        [new_version],
    ]
    rows = sorted(
        (
            r
            for r in observations
            if r["app"] == run["pool"]
            and len(r["replicas"]) == 8
            and all("error" not in n for n in r["replicas"])
        ),
        key=lambda r: r["time"],
    )
    assert rows

    def version(identity):
        return int(identity.rsplit("/weight_v", 1)[1])

    def identities(replica):
        return set(replica["recent"]) | set(replica["adapters"])

    members = {n["replica"] for n in rows[0]["replicas"]}
    previous = {n["replica"]: n["adapter_loads"] for n in rows[0]["replicas"]}
    for row in rows:
        assert {n["replica"] for n in row["replicas"]} == members
        for replica in row["replicas"]:
            assert replica["adapter_loads"] >= previous[replica["replica"]]
            previous[replica["replica"]] = replica["adapter_loads"]

    before = None
    for row in rows:
        if any(
            version(a) != first_version for n in row["replicas"] for a in identities(n)
        ):
            break
        # Account for every registered identity; a missing history could hide
        # an earlier registration of the new version.
        assert all(n["adapter_loads"] == len(identities(n)) for n in row["replicas"])
        before = row
    assert before is not None
    assert sum(n["adapter_loads"] for n in before["replicas"]) == sum(
        n["adapter_loads"] for n in rows[0]["replicas"]
    )
    after = rows[-1]
    assert all(n["active"] == 0 and not n["adapters"] for n in after["replicas"])
    assert all(n["adapter_loads"] == len(set(n["recent"])) for n in after["replicas"])
    registered = [a for n in after["replicas"] for a in n["recent"]]
    assert {version(a) for a in registered} == {first_version, new_version}
    new_copies = [a for a in registered if version(a) == new_version]
    assert len(set(new_copies)) == 16
    return {
        "new_version": new_version,
        "new_version_adapter_count": len(set(new_copies)),
        "new_version_replica_registrations": len(new_copies),
        "new_version_mean_replica_copies": len(new_copies) / len(set(new_copies)),
        "post_first_wave_registrations": sum(
            n["adapter_loads"] for n in after["replicas"]
        )
        - sum(n["adapter_loads"] for n in before["replicas"]),
        "post_first_wave_registration_call_seconds": sum(
            n["adapter_load_seconds"] for n in after["replicas"]
        )
        - sum(n["adapter_load_seconds"] for n in before["replicas"]),
        "before_observed_at": before["time"],
        "after_observed_at": after["time"],
        "measurement": "Complete final histories match registration counters on every replica. New-version copies are CPU registrations. RPC seconds are summed across replicas and include server waiting; they are not GPU transfer time or critical-path wall time. Wave zero and warmup are excluded from counter deltas.",
        "boundary_observations": {"before": before, "after": after},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--router-commit", help="Revision used by the measured workers")
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    raw = args.replay.read_bytes()
    runs = json.loads(raw)
    summary, rows = summarize(runs)
    if args.observations:
        observed = args.observations.read_bytes()
        observations = [json.loads(line) for line in observed.splitlines()]
        summary["observations_sha256"] = hashlib.sha256(observed).hexdigest()
        for run in runs:
            summary[run["mode"]]["replica_observations"] = summarize_observations(
                run, observations
            )
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
