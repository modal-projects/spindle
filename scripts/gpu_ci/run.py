"""Manual GPU CI: python -m scripts.gpu_ci.run prepare/run/cleanup/report."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import modal
from modal.client import _Client
from modal_proto import api_pb2

from scripts.gpu_ci.cleanup import owned_apps, stop_apps
from scripts.gpu_ci.diagnostics import analyze
from scripts.gpu_ci.validate import events, report
from spindle.control_plane.keys import placement_key
from spindle.providers.modal.kv import app_store_name

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
EXPECTED = ROOT / "tests/gpu/dapo_expected.json"


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(output, mode, environment):
    output.mkdir(parents=True, exist_ok=False)
    expected = json.loads(EXPECTED.read_text())
    cfg = json.loads((HERE / "config.json").read_text())
    cfg["updates"] = expected[f"{mode}_updates"]
    seconds = 3600 if mode == "correctness" else 10800
    deadline = int(time.time()) + seconds
    name = f"spindle-ci-dapo-{deadline}-{uuid.uuid4().hex[:12]}"
    plan = dict(
        name=name,
        apps=owned_apps(name),
        mode=mode,
        environment=environment,
        deadline=deadline,
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dataset_sha256=hashlib.sha256((HERE / "dapo.jsonl").read_bytes()).hexdigest(),
        stagger_seconds=0
        if mode == "correctness"
        else expected["performance_stagger_seconds"],
    )
    write(output / "plan.json", plan)
    write(output / "config.json", cfg)
    write(output / "expected.json", expected)
    overrides = {"platform.frontend": name, "platform.modal.environment": environment}
    # Environment credentials live in the dedicated Modal environment's named
    # secrets. No API keys or workstation resource names enter the recipe.
    (output / "deployment.py").write_text(
        "from scripts.gpu_ci.deployment import Config\n\n"
        + f"config = Config(name={name!r}, overrides={overrides!r})\n"
    )
    print(json.dumps(plan, indent=2))


async def topology(plan, model_ids):
    c = await _Client.from_env()
    response = await c.stub.TaskList(
        api_pb2.TaskListRequest(environment_name=plan["environment"])
    )
    tasks = []
    for task in response.tasks:
        if task.app_description not in plan["apps"]:
            continue
        info = (
            await c.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=task.task_id))
        ).info
        role = next(
            (
                role
                for role in ("trainer", "inference", "lora")
                if task.app_description == f"spindle-{role}-{plan['name']}"
            ),
            "frontend",
        )
        tasks.append(
            dict(
                task_id=task.task_id,
                role=role,
                started_at=info.started_at,
                finished_at=info.finished_at,
                gpu_type=info.gpu_type,
                gpu_count=info.gpu_config.count,
            )
        )
    app = await modal.App.lookup.aio(plan["name"], environment_name=plan["environment"])
    store = modal.Dict.from_name(
        app_store_name("spindle-models", app.app_id),
        environment_name=plan["environment"],
    )
    placements = {mid: await store.get.aio(placement_key(mid)) for mid in model_ids}
    return dict(tasks=tasks, placements=placements)


async def finalized_resources(output):
    path = output / "topology.json"
    if not path.exists():
        return
    tasks = json.loads(path.read_text())["tasks"]
    client = await _Client.from_env()
    deadline = time.monotonic() + 30
    while True:
        for row in tasks:
            info = (
                await client.stub.TaskGetInfo(
                    api_pb2.TaskGetInfoRequest(task_id=row["task_id"])
                )
            ).info
            row["started_at"], row["finished_at"] = info.started_at, info.finished_at
        gpus = [row for row in tasks if row["gpu_count"]]
        complete = bool(gpus) and all(
            row["finished_at"] > row["started_at"] > 0 for row in gpus
        )
        if complete or time.monotonic() >= deadline:
            write(
                output / "resources.json",
                {
                    "tasks": tasks,
                    "complete": complete,
                    "gpu_seconds": sum(
                        (row["finished_at"] - row["started_at"]) * row["gpu_count"]
                        for row in gpus
                    )
                    if complete
                    else None,
                    "notes": "Observed run tasks only; no dollar-price assumptions. Null GPU-seconds means final lifetimes were unavailable.",
                },
            )
            return
        await asyncio.sleep(2)


def collect_logs(plan, output):
    cleanup_path = output / "cleanup.json"
    stopped = (
        json.loads(cleanup_path.read_text())["stopped"] if cleanup_path.exists() else {}
    )
    for name in plan["apps"]:
        try:
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "modal",
                        "app",
                        "logs",
                        stopped.get(name, name),
                        "--env",
                        plan["environment"],
                        "--since",
                        "4h",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=15,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            pass  # Keep a partial log if fetching its full history takes too long.


def run(output):
    plan = json.loads((output / "plan.json").read_text())
    expected = json.loads((output / "expected.json").read_text())
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError(
            "TINKER_API_KEY must match spindle-api in the CI Modal environment"
        )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        "MODAL_ENVIRONMENT": plan["environment"],
        "SPINDLE_REQUEST_TIMING": "1",
    }
    children = []
    failure = None

    def interrupted(signum, frame):
        raise RuntimeError(f"CI interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(i, url):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.gpu_ci.client",
            "--config",
            str(output / "config.json"),
            "--output",
            str(output / f"client-{i:02d}"),
            "--base-url",
            url,
        ]
        if plan["mode"] == "correctness":
            cmd += ["--start-file", str(output / "START")]
        with (output / f"client-{i:02d}.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children.append(child)

    try:
        # Fail before GPU allocation unless the independent expiry backstop is
        # deployed and can execute with the CI environment's permissions.
        with (output / "deploy.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "deploy",
                    "-m",
                    "scripts.gpu_ci.cleanup",
                    "--env",
                    plan["environment"],
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=True,
            )
            modal.Function.from_name(
                "spindle-gpu-ci-janitor",
                "reap_expired",
                environment_name=plan["environment"],
            ).remote()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spindle.deployment_cli",
                    "deploy",
                    str(output / "deployment.py"),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=min(1200, max(1, plan["deadline"] - time.time())),
                check=True,
            )
        url = modal.Function.from_name(
            plan["name"], "server", environment_name=plan["environment"]
        ).get_web_url()
        if not url:
            raise RuntimeError("Deployment has no frontend URL")
        launch(0, url)
        anchor = None
        while True:
            if time.time() >= plan["deadline"] - 120:
                raise TimeoutError("CI deadline reached; reserving time for cleanup")
            traces = [
                events(output / f"client-{i:02d}" / "events.jsonl")
                for i in range(len(children))
            ]
            for i, child in enumerate(children):
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"Client {i} failed; see client log")
            if anchor is None and any(r["event"] == "ready" for r in traces[0]):
                anchor = time.monotonic()
            if anchor is not None and len(children) < expected["clients"]:
                if time.monotonic() >= anchor + len(children) * plan["stagger_seconds"]:
                    launch(len(children), url)
            if (
                plan["mode"] == "correctness"
                and len(traces) == expected["clients"]
                and all(any(r["event"] == "ready" for r in rows) for rows in traces)
            ):
                (output / "START").touch(exist_ok=True)
            write(
                output / "status.json",
                dict(
                    state="running",
                    heartbeat=time.time(),
                    updates=[
                        sum(r["event"] == "update_done" for r in rows)
                        for rows in traces
                    ],
                ),
            )
            if len(children) == expected["clients"] and all(
                c.poll() == 0 for c in children
            ):
                break
            time.sleep(2)
        traces = [
            events(output / f"client-{i:02d}" / "events.jsonl")
            for i in range(expected["clients"])
        ]
        mids = [
            next(r["model_id"] for r in rows if r["event"] == "client_created")
            for rows in traces
        ]
        write(
            output / "topology.json",
            asyncio.run(asyncio.wait_for(topology(plan, mids), timeout=120)),
        )
        report(output)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        write(output / "failure.json", {"error": failure})
        raise
    finally:
        # A second termination must not interrupt cleanup; the remote expiry
        # job still covers SIGKILL, machine loss, or a forced Actions timeout.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            cleanup(output)
        except Exception as exc:
            failure = failure or f"Cleanup failed: {exc}"
            write(output / "failure.json", {"error": failure})
            raise
        finally:
            try:
                collect_logs(plan, output)
                asyncio.run(asyncio.wait_for(finalized_resources(output), timeout=60))
                analyze(output)
            except Exception as exc:
                write(output / "diagnostics-error.json", {"error": str(exc)})
            write(
                output / "status.json",
                dict(
                    state="failed" if failure else "completed",
                    error=failure,
                    finished=time.time(),
                ),
            )


def cleanup(output):
    if not (output / "plan.json").exists():
        return
    plan = json.loads((output / "plan.json").read_text())
    stopped = asyncio.run(
        asyncio.wait_for(stop_apps(plan["environment"], plan["name"]), timeout=120)
    )
    write(output / "cleanup.json", dict(stopped=stopped, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "cleanup", "report"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["correctness", "performance"], default="correctness"
    )
    parser.add_argument("--environment", default="spindle-ci")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(output, args.mode, args.environment)
    elif args.action == "run":
        run(output)
    elif args.action == "cleanup":
        cleanup(output)
    else:
        print(json.dumps(report(output), indent=2))


if __name__ == "__main__":
    main()
