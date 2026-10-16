"""Launch independent copies of client.py and collect experiment resource lifetimes."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from modal._utils.async_utils import synchronize_api
from modal.client import _Client
from modal_proto import api_pb2

from scripts.benchmarks.lora_routing.recovery import classify_failure, latest_commit

ROOT = Path(__file__).resolve().parent


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def events(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(json.loads(line))
        except ValueError:
            pass
    return values


@synchronize_api
async def resources(environment, owned, known, since=0.0):
    client = await _Client.from_env()
    listed = await client.stub.TaskList(
        api_pb2.TaskListRequest(environment_name=environment)
    )
    task_apps = {t.task_id: t.app_description for t in listed.tasks}
    ids = set(known)
    ids.update(t.task_id for t in listed.tasks if t.app_description in owned)
    for task_id in ids:
        info = (
            await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=task_id))
        ).info
        if info.finished_at and info.finished_at < since:
            known.pop(task_id, None)
            continue
        known[task_id] = dict(
            task_id=task_id,
            app=task_apps.get(task_id, known.get(task_id, {}).get("app")),
            started_at=info.started_at,
            finished_at=info.finished_at,
            gpu_type=info.gpu_type,
            gpu_count=info.gpu_config.count,
        )
    return known


def interrupted_trainer(known, started, updates, expected_updates):
    if sum(updates) >= expected_updates:
        return None
    return next(
        (
            row["task_id"]
            for row in known.values()
            if row["gpu_count"] > 1 and row["finished_at"] > started
        ),
        None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--clients", type=int, default=16)
    p.add_argument("--stagger-seconds", type=float, default=5.0)
    p.add_argument("--hours", type=float, default=0.5)
    p.add_argument("--initialization-timeout", type=float, default=1200)
    p.add_argument(
        "--require-stable-trainer",
        action="store_true",
        help="Fail an evaluation if its multinode/multi-GPU trainer exits before all updates",
    )
    p.add_argument("--recover-clients", action="store_true")
    p.add_argument("--environment", default="lilo-deploy")
    p.add_argument("--owned-apps", nargs="+", required=True)
    p.add_argument(
        "--initialization-model",
        help="Release this prepared model after the first measured client attaches",
    )
    p.add_argument(
        "--resource-since",
        type=float,
        default=0.0,
        help="Exclude containers that finished before this experiment timestamp",
    )
    p.add_argument(
        "--keep-apps",
        action="store_true",
        help="Keep successful pilot deployments for the next stage",
    )
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    write(args.root / "config.json", config)
    state = dict(
        state="starting",
        pid=os.getpid(),
        started=time.time(),
        clients=args.clients,
        stagger_seconds=args.stagger_seconds,
        owned_apps=args.owned_apps,
    )
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    processes = []
    known = {}
    anchor = None
    last_collect = 0
    initialization_model = args.initialization_model
    error = None
    recovery_due = {}
    failures_by_step = {}
    handled_trainers = set()

    def launch(index):
        output = args.root / f"client-{index:02d}"
        failure = output / "failure.json"
        if failure.exists():
            failure.rename(output / f"failure-{time.time_ns()}.json")
        with (args.root / f"client-{index:02d}.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "client.py"),
                    "--config",
                    str(args.root / "config.json"),
                    "--output",
                    str(output),
                    "--base-url",
                    args.base_url,
                    "--client-index",
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        if index == len(processes):
            processes.append(child)
        else:
            processes[index] = child
        state.setdefault("launches", []).append(
            dict(client=index, pid=child.pid, time=time.time())
        )

    def recover(index, message):
        child = processes[index]
        classification = classify_failure(message)
        commit = latest_commit(args.root / f"client-{index:02d}")
        update = commit["update"] if commit else 0
        key = (index, update)
        failures_by_step[key] = failures_by_step.get(key, 0) + 1
        record = dict(
            client=index,
            time=time.time(),
            error=message,
            classification=classification,
            committed_update=update,
            checkpoint=commit["checkpoint_path"] if commit else None,
            retries_at_update=failures_by_step[key],
        )
        state.setdefault("recoveries", []).append(record)
        write(args.root / "recovery.json", state["recoveries"])
        if classification != "retry_from_checkpoint" or failures_by_step[key] > 3:
            raise RuntimeError(f"Client {index}: {classification}; {message}")
        # Kill the old process before creating a replacement; never replay a
        # possibly-applied optimizer operation on the old training session.
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        trace = events(args.root / f"client-{index:02d}" / "events.jsonl")
        models = [e["model_id"] for e in trace if e["event"] == "client_created"]
        if models:
            try:
                response = httpx.post(
                    args.base_url + "/api/v1/unload_model",
                    headers={"x-api-key": os.environ["TINKER_API_KEY"]},
                    json={"model_id": models[-1]},
                    timeout=30,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                record["unload_error"] = str(exc)
        recovery_due[index] = time.time() + min(
            300, 30 * 2 ** (failures_by_step[key] - 1)
        )
        record["retry_at"] = recovery_due[index]
        write(args.root / "recovery.json", state["recoveries"])

    try:
        launch(0)
        while True:
            now = time.time()
            if stop or (args.root / "STOP_NOW").exists():
                raise RuntimeError("Stop requested")
            if anchor is None and now - state["started"] > args.initialization_timeout:
                raise TimeoutError("First training client did not become ready")
            if now - state["started"] > args.hours * 3600:
                raise TimeoutError("Experiment deadline")
            traces = [
                events(args.root / f"client-{i:02d}" / "events.jsonl")
                for i in range(len(processes))
            ]
            if initialization_model and any(
                r["event"] == "client_created" for r in traces[0]
            ):
                response = httpx.post(
                    args.base_url + "/api/v1/unload_model",
                    headers={"x-api-key": os.environ["TINKER_API_KEY"]},
                    json={"model_id": initialization_model},
                    timeout=120,
                )
                response.raise_for_status()
                state["initialization_model_released_at"] = time.time()
                initialization_model = None
            if anchor is None and any(r["event"] == "ready" for r in traces[0]):
                anchor = time.monotonic()
                state["stagger_started_at"] = now
            if (
                anchor is not None
                and len(processes) < args.clients
                and time.monotonic() >= anchor + len(processes) * args.stagger_seconds
            ):
                launch(len(processes))
            trainer_failure = args.root / "RECOVER_TRAINER.json"
            lost_trainer = None
            if args.recover_clients and trainer_failure.exists():
                notice = json.loads(trainer_failure.read_text())
                if notice["trainer"] not in handled_trainers:
                    lost_trainer = notice["trainer"]
                    handled_trainers.add(lost_trainer)
            for i, child in enumerate(processes):
                if i in recovery_due:
                    if now >= recovery_due[i]:
                        launch(i)
                        del recovery_due[i]
                    continue
                failure = args.root / f"client-{i:02d}" / "failure.json"
                message = None
                if failure.exists():
                    message = json.loads(failure.read_text())["error"]
                elif lost_trainer and child.poll() != 0:
                    message = f"Trainer exited: {lost_trainer}"
                elif child.poll() not in (None, 0):
                    message = f"Client process terminated with exit {child.returncode}"
                if message:
                    if args.recover_clients:
                        recover(i, message)
                    else:
                        raise RuntimeError(f"Client {i}: {message}")
            state.update(
                state="running" if anchor else "initializing",
                heartbeat=now,
                updates=[
                    len(list((args.root / f"client-{i:02d}").glob("commit-*.json")))
                    if config.get("checkpoint_each_update", False)
                    else len(
                        {
                            r["update"]
                            for r in events(
                                args.root / f"client-{i:02d}" / "events.jsonl"
                            )
                            if r["event"] == "update_done"
                        }
                    )
                    for i in range(len(processes))
                ],
            )
            write(args.root / "status.json", state)
            if now - last_collect > 30:
                last_collect = now
                try:
                    known = asyncio.run(
                        asyncio.wait_for(
                            resources.aio(
                                args.environment,
                                args.owned_apps,
                                known,
                                args.resource_since,
                            ),
                            timeout=20,
                        )
                    )
                    write(args.root / "gpu-lifetimes.json", list(known.values()))
                except Exception as exc:
                    state["resource_error"] = str(exc)
            if args.require_stable_trainer:
                interrupted = interrupted_trainer(
                    known,
                    state["started"],
                    state["updates"],
                    args.clients * config["updates"],
                )
                if interrupted:
                    raise RuntimeError(
                        f"Trainer {interrupted} exited before all updates"
                    )
            if (
                not recovery_due
                and len(processes) == args.clients
                and all(c.poll() == 0 for c in processes)
            ):
                if state["updates"] != [config["updates"]] * args.clients:
                    raise RuntimeError(
                        "Client exited without committing all required updates"
                    )
                break
            time.sleep(5)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for child in processes:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in processes:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state["cleanup"] = {}
        for app in [] if args.keep_apps and error is None else args.owned_apps:
            try:
                r = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "modal",
                        "app",
                        "stop",
                        "-y",
                        app,
                        "--env",
                        args.environment,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                state["cleanup"][app] = dict(
                    returncode=r.returncode, output=(r.stdout + r.stderr)[-500:]
                )
            except Exception as exc:
                state["cleanup"][app] = dict(error=str(exc))
        try:
            if args.keep_apps and error is None:
                state.update(state="completed", error=None, finished=time.time())
                write(args.root / "status.json", state)
                return
            drain_deadline = time.monotonic() + 180
            while True:
                known = asyncio.run(
                    asyncio.wait_for(
                        resources.aio(
                            args.environment,
                            args.owned_apps,
                            known,
                            args.resource_since,
                        ),
                        timeout=20,
                    )
                )
                write(args.root / "gpu-lifetimes.json", list(known.values()))
                running = [
                    r
                    for r in known.values()
                    if r["gpu_count"] and r["started_at"] and not r["finished_at"]
                ]
                if not running or time.monotonic() >= drain_deadline:
                    break
                time.sleep(5)
        except Exception as exc:
            state["resource_error"] = str(exc)
        state.update(
            state="failed" if error else "completed", error=error, finished=time.time()
        )
        write(args.root / "status.json", state)
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
