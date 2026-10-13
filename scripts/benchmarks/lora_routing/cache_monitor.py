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
    args = parser.parse_args()
    owned = sorted(
        {p["pool"] for p in json.loads((args.root / "pools.json").read_text())}
    )
    assert 1 <= len(owned) <= 3 and all(
        p.startswith("spindle-lora-cache27-") for p in owned
    )
    gpu_limit = 8 * len(owned)
    known = {}
    observed_peak = 0
    deadline = time.time() + args.minutes * 60
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
