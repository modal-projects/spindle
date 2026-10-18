"""Compare the predeclared later-order one-batch control with primary trials.

This is a supplementary comparison, not an additional independent replication.
The physical replicas, trace specification, calibration and policy sources must
match. A later execution order remains a limitation even when those checks pass.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from confirmation_report import METRICS
from paper_results import analyzed
from supervise import write


def matched_pair(candidate, baseline, frozen):
    for raw in (candidate, baseline):
        if raw.get("error"):
            raise ValueError("Interrupted arm cannot enter a paired comparison")
        for name, digest in raw["config"]["source_sha256"].items():
            allowed = frozen.get("accepted_harness_source_sha256", {}).get(
                name, [frozen["source_sha256"][name]]
            )
            if digest not in allowed:
                raise ValueError(f"Unapproved source: {name}")
    for field in ("pool", "workload_sha256", "trace_sha256"):
        if candidate[field] != baseline[field]:
            raise ValueError(f"Unmatched {field}")
    for field in ("work_rates", "source_sha256"):
        if candidate["config"][field] != baseline["config"][field]:
            raise ValueError(f"Unmatched {field}")
    incarnations = [
        {state["replica"] for state in raw[boundary]}
        for raw in (candidate, baseline)
        for boundary in ("baseline", "final")
    ]
    if len(incarnations[0]) != 8 or any(s != incarnations[0] for s in incarnations):
        raise ValueError("Physical replicas changed between or within arms")


def summarize(frozen, roots):
    pairs, missing, failures, values = {}, [], {}, {}
    for job in frozen["jobs"]:
        primary = f"confirm-{job['family']}-r{job['trial']}"
        folders = [root for root in roots if (root / job["name"]).exists()]
        if len(folders) > 1:
            raise ValueError(f"Duplicate supplementary job: {job['name']}")
        if not folders:
            missing.append(job["name"])
            continue
        paths = {
            frozen["candidate"]: folders[0] / primary / f"{frozen['candidate']}.json",
            "model-wave": folders[0] / job["name"] / "model-wave.json",
        }
        absent = [str(path) for path in paths.values() if not path.exists()]
        if absent:
            missing.extend(absent)
            continue
        raw = {policy: json.loads(path.read_text()) for policy, path in paths.items()}
        failed = {
            policy: data["error"] for policy, data in raw.items() if data.get("error")
        }
        if failed:
            failures[job["name"]] = failed
            continue
        matched_pair(raw[frozen["candidate"]], raw["model-wave"], frozen)
        rows = {}
        for policy, data in raw.items():
            row = analyzed(data)
            attempts = [a["attrs"] for r in data["requests"] for a in r["attempts"]]
            row.update(
                raw_path=str(paths[policy]),
                raw_sha256=hashlib.sha256(paths[policy].read_bytes()).hexdigest(),
                workload_sha256=data["workload_sha256"],
                transport_failures=sum("error_type" in a for a in attempts),
                native_destination_attempts=sum(
                    a.get("selected_replica") is None for a in attempts
                ),
            )
            rows[policy] = row
        pairs[job["name"]] = rows
        for metric in METRICS:
            a, b = rows[frozen["candidate"]][metric], rows["model-wave"][metric]
            if a is not None and b not in (None, 0):
                values.setdefault(f"{job['family']}/{metric}", []).append(
                    dict(trial=job["trial"], candidate=a, baseline=b, ratio=a / b)
                )
    estimates = {}
    for key, rows in values.items():
        ratios = [row["ratio"] for row in rows]
        result = dict(
            trials=rows,
            n=len(rows),
            minimum_ratio=min(ratios),
            maximum_ratio=max(ratios),
        )
        if all(r > 0 for r in ratios):
            logs = [math.log(r) for r in ratios]
            mean = statistics.mean(logs)
            result["geometric_mean_ratio"] = math.exp(mean)
            if len(rows) == frozen["trials"]:
                critical = {3: 4.302652729696142, 5: 2.7764451051977987}[len(rows)]
                radius = critical * statistics.stdev(logs) / math.sqrt(len(rows))
                result["geometric_mean_ratio_95pct_interval"] = [
                    math.exp(mean - radius),
                    math.exp(mean + radius),
                ]
        estimates[key] = result
    return dict(
        complete=not missing and not failures,
        missing=missing,
        failures=failures,
        pairs=pairs,
        estimates=estimates,
        interpretation=frozen["interpretation"],
        uncertainty="Same three primary trial seeds; supplementary arms always run later. Trial-level log-ratio t intervals do not remove order or drift effects. Requests and controls are not additional independent trials.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.freeze.read_text()), args.roots)
    write(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "pairs"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
