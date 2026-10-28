"""Start independent DAPO clients. There is no cross-client step barrier.

The blog two-client run uses --mode ready and a 150 second stagger: client 1
starts 150 seconds after client 0 logs ready. The utilization sweep uses
--mode gate: every client finishes warmup, then rollouts are offset by
--stagger-seconds from one shared start time.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def steps_config(path: Path, steps: int | None) -> Path:
    if steps is None:
        return path
    if steps < 1:
        raise SystemExit("steps must be positive")
    cfg = json.loads(path.read_text())
    cfg["updates"] = steps
    dest = Path(tempfile.mkdtemp()) / "config.json"
    dest.write_text(json.dumps(cfg))
    return dest


def ready(path):
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        try:
            if json.loads(line).get("event") == "ready":
                return True
        except json.JSONDecodeError:
            continue
    return False


def spawn(index, config, base_url, start_gate):
    output = ROOT / f"client-{index:02d}"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "client.py"),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if base_url:
        command.extend(["--base-url", base_url])
    if start_gate is not None:
        command.extend(["--start-gate", str(start_gate)])
    log = (ROOT / f"client-{index:02d}.log").open("ab")
    return subprocess.Popen(
        command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--stagger-seconds", type=float, default=150)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--steps", type=int, help="override config updates")
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--mode", choices=("ready", "gate"), default="ready")
    args = parser.parse_args()
    if args.clients < 1:
        raise SystemExit("clients must be positive")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY")
    config = steps_config(args.config, args.steps)
    processes = []
    if args.mode == "gate":
        gate = ROOT / "start.json"
        if gate.exists():
            raise SystemExit(
                f"{gate} already exists; remove it before a new sweep stage"
            )
        for index in range(args.clients):
            processes.append(spawn(index, config, args.base_url, gate))
        while not all(
            ready(ROOT / f"client-{i:02d}" / "events.jsonl")
            for i in range(args.clients)
        ):
            if any(process.poll() not in (None, 0) for process in processes):
                raise SystemExit("A client exited before warmup finished")
            time.sleep(2)
        gate.write_text(
            json.dumps(
                {
                    "time": time.time() + args.stagger_seconds,
                    "stagger_seconds": args.stagger_seconds,
                }
            )
            + "\n"
        )
    else:
        for index in range(args.clients):
            processes.append(spawn(index, config, args.base_url, None))
            while not ready(ROOT / f"client-{index:02d}" / "events.jsonl"):
                if processes[-1].poll() not in (None, 0):
                    raise SystemExit(f"Client {index} exited before it was ready")
                time.sleep(2)
            if index + 1 < args.clients:
                time.sleep(args.stagger_seconds)
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"Client exit codes: {codes}")


if __name__ == "__main__":
    main()
