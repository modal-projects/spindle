"""Render client curves and backend occupancy from collected logs, when available."""

import json
import statistics

from scripts.gpu_ci.validate import events


def merge_seconds(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def analyze(root):
    plan = json.loads((root / "plan.json").read_text())
    clients = [events(p) for p in sorted(root.glob("client-*/events.jsonl"))]
    started = [
        r["time"] for rows in clients for r in rows if r["event"] == "training_started"
    ]
    ended = [
        r["time"] for rows in clients for r in rows if r["event"] == "training_finished"
    ]
    if not started or not ended:
        return {"status": "incomplete"}
    start, end = min(started), max(ended)
    phases = []
    requests = {}
    log = root / f"spindle-trainer-{plan['name']}.log"
    for line in log.read_text().splitlines() if log.exists() else []:
        for event in ("spindle_step_timing", "spindle_request_mark"):
            i = line.find('{"event": "' + event + '"')
            if i < 0:
                continue
            try:
                r = json.JSONDecoder().raw_decode(line[i:])[0]
            except json.JSONDecodeError:
                continue
            if not start <= r["ts"] <= end:
                continue
            if event == "spindle_step_timing":
                if r["phase"] in (
                    "forward_backward",
                    "optim_step",
                    "save_sampler_weights",
                ):
                    phases.append((max(start, r["ts"] - r["seconds"]), r["ts"]))
            else:
                for rid in r.get("request_ids", [r.get("request_id")]):
                    if rid:
                        requests.setdefault(rid, {})[r["mark"]] = r
    waits = []
    for rows in clients:
        model = next(r["model_id"] for r in rows if r["event"] == "client_created")
        for done in (r for r in rows if r["event"] == "train_done"):
            begin = next(
                r["time"]
                for r in rows
                if r["event"] == "train_start" and r["update"] == done["update"]
            )
            group = [
                r
                for rid, r in requests.items()
                if rid.startswith(model + ":")
                and "engine.op.exec_begin" in r
                and r["engine.op.exec_begin"]["kind"] == "forward_backward"
                and begin <= r["engine.op.exec_begin"]["ts"] < done["time"]
            ]
            if not group or any("engine.op.submitted" not in r for r in group):
                continue
            starts = {r["engine.op.exec_begin"]["ts"] for r in group}
            # A split execution cannot be represented by one queue interval.
            if len(starts) != 1:
                continue
            queue = min(starts) - max(r["engine.op.submitted"]["ts"] for r in group)
            if queue >= 0:
                waits.append(queue)
    result = dict(
        status="available" if phases else "missing_backend_logs",
        backend_call_occupancy=merge_seconds(phases) / (end - start)
        if phases
        else None,
        training_queue_mean_s=statistics.mean(waits) if waits else None,
        training_queue_observations=len(waits),
        notes="Queue = last chunk accepted to execution start for updates executed in one batch. Coverage may be partial. Backend occupancy is host call time, not GPU hardware utilization; log collection may be incomplete.",
    )
    (root / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
