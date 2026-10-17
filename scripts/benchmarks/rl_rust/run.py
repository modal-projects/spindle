"""Run a matched pair through the existing recoverable independent-client harness."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.benchmarks.lora_routing.recovery import atomic_json
from scripts.benchmarks.lora_routing.supervise import resources
from scripts.benchmarks.rl_four_arm.guard import inspect
from scripts.benchmarks.rl_four_arm.prepare_holders import prepare
from spindle.deployments import DeploymentConfig

REPO = Path(__file__).resolve().parents[3]


def policies(root):
    return json.loads((root / "schedule.json").read_text())["pairs"][0]


def command(module, *args, **kwargs):
    return subprocess.run(
        [sys.executable, "-u", "-m", module, *map(str, args)],
        check=True,
        cwd=REPO,
        **kwargs,
    )


def validate_inputs(root):
    keys = (
        "config_sha256",
        "prompts_sha256",
        "dataset_sha256",
        "client_index",
        "problem_offset",
        "stop_token_ids",
        "packages",
        "source_sha256",
    )
    for index in range(8):
        paths = [
            root / p / "measured" / f"client-{index:02d}" / "manifest.json"
            for p in policies(root)
        ]
        if all(path.exists() for path in paths):
            a, *others = [json.loads(path.read_text()) for path in paths]
            if any(a[k] != b[k] for b in others for k in keys):
                raise RuntimeError(f"Mismatched paired inputs for client {index}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    os.environ["TINKER_API_KEY"] = Path("/tmp/spindle-rl-three-arm-api-key").read_text()
    config = json.loads((root / "config.json").read_text())
    owned = sum(
        (
            json.loads((root / p / "owned-apps.json").read_text())
            for p in policies(root)
        ),
        [],
    )
    external = json.loads((args.external_root / "owned-apps.json").read_text())
    (root / "owned-apps.json").write_text(json.dumps(owned))
    source_files = list((REPO / "src").rglob("*.py")) + list(
        (REPO / "rust/router/src").glob("*.rs")
    )
    source_files += list((REPO / "scripts/benchmarks").rglob("*.py"))
    atomic_json(
        root / "source-manifest.json",
        {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
    )
    state = dict(started=time.time(), phase="initializing", complete=False)
    known = {}
    collector = runner = None
    pool = ThreadPoolExecutor(len(policies(root)))
    ready = {p: pool.submit(prepare, root, p, config) for p in policies(root)}
    monitor_failures = 0
    collector_restarts = 0
    trainers = {}
    try:
        while True:
            if (root / "STOP").exists():
                raise RuntimeError("Experiment stop requested")
            try:
                known = asyncio.run(
                    asyncio.wait_for(
                        resources.aio(
                            "lilo-deploy", owned + external, known, state["started"]
                        ),
                        30,
                    )
                )
                monitor_failures = 0
            except Exception:
                monitor_failures += 1
                if monitor_failures >= 3:
                    raise
                time.sleep(5)
                continue
            active = sum(r["gpu_count"] for r in known.values() if not r["finished_at"])
            if active > 48:
                raise RuntimeError(f"Combined experiment GPU limit exceeded: {active}")
            atomic_json(root / "gpu-lifetimes.json", list(known.values()))
            state.update(heartbeat=time.time(), active_gpus_including_existing=active)
            for policy, future in ready.items():
                if future.done():
                    future.result()
            if runner is None and all(f.done() for f in ready.values()):
                for p in policies(root):
                    deployment = json.loads((root / p / "deployment.json").read_text())
                    # Early manifests included this derived property.
                    deployment.pop("trainer_app_name", None)
                    app = DeploymentConfig.model_validate(deployment).trainer_app_name
                    candidates = [
                        r
                        for r in known.values()
                        if r["app"] == app and not r["finished_at"] and r["gpu_count"]
                    ]
                    if len(candidates) != 1:
                        raise RuntimeError(f"Expected one serving trainer for {p}")
                    trainers[p] = candidates[0]["task_id"]
                atomic_json(root / "trainer-incarnations.json", trainers)
                with (root / "run.log").open("a") as out:
                    runner = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.benchmarks.rl_four_arm.run",
                            "--root",
                            str(root),
                            *[arg for p in policies(root) for arg in ("--policy", p)],
                        ],
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                state.update(phase="training", measured_started=time.time())
            if runner is not None:
                validate_inputs(root)
                state["updates"] = inspect(root, trainers)
                if runner.poll() is not None:
                    if runner.returncode:
                        raise RuntimeError(
                            f"Measured pair exited {runner.returncode}; see per-client failure records"
                        )
                    if not all(n == 80 for n in state["updates"].values()):
                        raise RuntimeError("Incomplete update count")
                    state.update(complete=True, phase="complete")
                    break
                if collector is None or collector.poll() is not None:
                    collector_restarts += 1
                    if collector_restarts > 5:
                        raise RuntimeError("Repeated observation collector failure")
                    with (root / "collector.log").open("a") as out:
                        collector = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                "-m",
                                "scripts.benchmarks.rl_rust.collect",
                                "--root",
                                str(root),
                            ],
                            stdout=out,
                            stderr=subprocess.STDOUT,
                        )
            atomic_json(root / "status.json", state)
            if time.time() - state["started"] > 12 * 3600:
                raise TimeoutError("12-hour experiment deadline")
            time.sleep(15)
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["finished"] = time.time()
        atomic_json(root / "status.json", state)
        (root / "STOP").write_text(state.get("error", "Completed"))
        for p in policies(root):
            directory = root / p / "measured"
            if directory.exists():
                (directory / "STOP_NOW").write_text("Pair ended")
        for child in (runner, collector):
            if child is not None and child.poll() is None:
                child.terminate()
        for app in owned:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "app",
                    "stop",
                    "-y",
                    app,
                    "--env",
                    "lilo-deploy",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        settle_deadline = time.monotonic() + 300
        while True:
            known = asyncio.run(
                resources.aio("lilo-deploy", owned, known, state["started"])
            )
            remaining = [
                r
                for r in known.values()
                if r.get("app") in owned and r["gpu_count"] and not r["finished_at"]
            ]
            if not remaining:
                break
            if time.monotonic() >= settle_deadline:
                raise RuntimeError(
                    "Owned GPUs still active after cleanup; do not start next pair"
                )
            time.sleep(10)
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
