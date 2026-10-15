"""Bound the lifetime of only this experiment's pools and audit GPU allocation."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from supervise import resources, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument(
        "--deadline", type=float, help="Absolute UTC epoch shutdown time"
    )
    parser.add_argument("--owned-apps", type=Path, help="Explicit live-RL app manifest")
    parser.add_argument("--max-gpus", type=int, default=16)
    args = parser.parse_args()
    if args.owned_apps:
        owned = json.loads(args.owned_apps.read_text())
        assert 1 <= len(owned) <= 4 and all(
            p.startswith(
                (
                    "spindle-routing27-live-",
                    "spindle-trainer-routing27-live-",
                    "spindle-inference-routing27-live-",
                    "spindle-lora-routing27-live-",
                )
            )
            for p in owned
        )
        assert 1 <= args.max_gpus <= 48
        gpu_limit = args.max_gpus
    else:
        owned = sorted(
            {p["pool"] for p in json.loads((args.root / "pools.json").read_text())}
        )
        assert 1 <= len(owned) <= 3 and all(
            p.startswith("spindle-lora-cache27-") for p in owned
        )
        deployments = {
            p["pool"]: p["deployment"]["recipe"]
            for p in json.loads((args.root / "pools.json").read_text())
        }
        gpu_limit = min(
            args.max_gpus,
            sum(
                r["inference_max_replicas"] * r["inference_gpus_per_node"]
                for r in deployments.values()
            ),
        )
    lifetimes = args.root / "gpu-lifetimes.json"
    known = (
        {r["task_id"]: r for r in json.loads(lifetimes.read_text())}
        if lifetimes.exists()
        else {}
    )
    observed_peak = 0
    deadline = (
        args.deadline if args.deadline is not None else time.time() + args.minutes * 60
    )
    error = None
    try:
        while time.time() < deadline and not (args.root / "STOP").exists():
            known = resources("lilo-deploy", owned, known)
            write(args.root / "gpu-lifetimes.json", list(known.values()))
            active = sum(r["gpu_count"] for r in known.values() if not r["finished_at"])
            observed_peak = max(observed_peak, active)
            write(
                args.root / "monitor.json",
                {"active_gpus": active, "updated": time.time(), "deadline": deadline},
            )
            if active > gpu_limit:
                raise RuntimeError(
                    f"Experiment exceeded its {gpu_limit}-GPU allocation: {active}"
                )
            time.sleep(15)
    except Exception as exc:
        error = str(exc)
    finally:
        for pool in owned:
            subprocess.run(
                ["modal", "app", "stop", "-y", pool, "--env", "lilo-deploy"],
                check=False,
                timeout=120,
            )
        for _ in range(24):
            known = resources("lilo-deploy", owned, known)
            write(args.root / "gpu-lifetimes.json", list(known.values()))
            if not any(r["gpu_count"] and not r["finished_at"] for r in known.values()):
                break
            time.sleep(5)
        points = []
        for r in known.values():
            if r["gpu_count"] and r["started_at"]:
                points.append((r["started_at"], r["gpu_count"]))
                if r["finished_at"]:
                    points.append((r["finished_at"], -r["gpu_count"]))
        active = peak = 0
        for _, delta in sorted(points):
            active += delta
            peak = max(peak, active)
        active = sum(r["gpu_count"] for r in known.values() if not r["finished_at"])
        write(
            args.root / "resource-audit.json",
            {
                "active_gpus": active,
                "peak_gpus": max(peak, observed_peak),
                "error": error,
                "finished": time.time(),
            },
        )


if __name__ == "__main__":
    main()
