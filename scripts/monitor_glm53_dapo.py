"""Follow a GLM Modal run and save client progress, memory peaks and errors."""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--env", default="kailash-dev")
    parser.add_argument("--stop-on-oom", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "app.log"
    log_path.touch()
    seen = set()
    clients = {}
    completed = {}
    validation = {}
    decoding = {}
    errors = []
    max_allocated = 0
    max_reserved = 0
    validation_passed = False
    validation_passed_at = None
    last_progress = time.time()
    last_status = 0
    app = None
    follower = None
    partial = ""
    modal = [sys.executable, "-m", "modal"]
    with log_path.open("a") as output, log_path.open() as stream:
        try:
            while True:
                if follower is None or follower.poll() is not None:
                    follower = subprocess.Popen(
                        modal
                        + [
                            "app",
                            "logs",
                            args.app_id,
                            "--env",
                            args.env,
                            "--follow",
                            "--timestamps",
                            "--show-container-id",
                        ],
                        stdout=output,
                        stderr=output,
                    )
                partial += stream.read()
                lines = partial.split("\n")
                partial = lines.pop()
                for line in lines:
                    if line in seen:
                        continue
                    seen.add(line)
                    if "GLM VALIDATION PASSED:" in line:
                        if not validation_passed:
                            validation_passed_at = datetime.fromisoformat(
                                " ".join(line.split(" ")[:2])
                            ).timestamp()
                            clients = {}
                        validation_passed = True
                        last_progress = time.time()
                    if any(
                        s in line
                        for s in (
                            "Traceback (most recent call last)",
                            "OutOfMemoryError",
                            "CUDA out of memory",
                            "RuntimeError:",
                            "AssertionError:",
                            "DAPO STATUS",
                        )
                    ):
                        if "DAPO STATUS" not in line or '"phase": "failed"' in line:
                            errors.append(line[-2500:])
                            errors[:] = errors[-30:]
                    if '"client":' in line and "{" in line:
                        try:
                            row, _ = json.JSONDecoder().raw_decode(
                                line[line.index("{") :]
                            )
                        except ValueError:
                            continue
                        if "client" not in row or "step" not in row:
                            continue
                        client = row["client"]
                        if "phase" in row and "time" in row:
                            if (
                                validation_passed_at is not None
                                and row["time"] < validation_passed_at
                            ):
                                continue
                            state = clients.setdefault(client, {})
                            lane = row.get("lane", "update")
                            if row["time"] >= state.get(lane, {}).get("time", 0):
                                state[lane] = row
                        elif "raw_rollouts" in row:
                            target = (
                                completed
                                if validation_passed_at is not None
                                and row["end_time"] >= validation_passed_at
                                else validation
                            )
                            target[client, row["step"]] = row
                        last_progress = time.time()
                    match = re.search(
                        r"max_allocated_gb=([\d.]+).*max_reserved_gb=([\d.]+)", line
                    )
                    if match:
                        max_allocated = max(max_allocated, float(match[1]))
                        max_reserved = max(max_reserved, float(match[2]))
                        last_progress = time.time()
                    match = re.search(
                        r"#running-req: (\d+).*cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+)",
                        line,
                    )
                    if match:
                        fields = line.split(" ", 3)
                        decoding[fields[2]] = dict(
                            time=" ".join(fields[:2]),
                            active_requests=int(match[1]),
                            cuda_graph=match[2] == "True",
                            tokens_per_second=float(match[3]),
                        )
                        last_progress = time.time()
                if time.time() - last_status >= 300:
                    status = subprocess.run(
                        modal + ["app", "list", "--env", args.env, "--json"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if status.returncode == 0:
                        app = next(
                            (
                                r
                                for r in json.loads(status.stdout)
                                if r["app_id"] == args.app_id
                            ),
                            None,
                        )
                    last_status = time.time()
                warnings = []
                if time.time() - last_progress > 1800:
                    warnings.append(
                        "No new client, decode or memory progress for 30 minutes"
                    )
                if any(not row["cuda_graph"] for row in decoding.values()):
                    warnings.append("A replica last reported eager decode")
                state = dict(
                    app_id=args.app_id,
                    app=app,
                    checked_at=datetime.now(timezone.utc).isoformat(),
                    validation_passed=validation_passed,
                    validation_steps=list(validation.values()),
                    completed_steps=list(completed.values()),
                    clients=clients,
                    replicas=decoding,
                    max_allocated_gib=max_allocated,
                    max_reserved_gib=max_reserved,
                    errors=errors,
                    warnings=warnings,
                )
                stopped_for_oom = False
                if args.stop_on_oom and any(
                    "OutOfMemoryError" in error or "CUDA out of memory" in error
                    for error in errors
                ):
                    stopped = subprocess.run(
                        modal
                        + ["app", "stop", args.app_id, "--env", args.env, "--yes"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    stopped_for_oom = stopped.returncode == 0
                    state["stopped_for_oom"] = stopped_for_oom
                    state["oom_stop_error"] = stopped.stderr[-2000:]
                temp = args.output / "health.tmp"
                temp.write_text(json.dumps(state, indent=2) + "\n")
                temp.replace(args.output / "health.json")
                if stopped_for_oom or (app is not None and app.get("stopped_at")):
                    break
                time.sleep(15)
        finally:
            if follower is not None:
                follower.terminate()


if __name__ == "__main__":
    main()
