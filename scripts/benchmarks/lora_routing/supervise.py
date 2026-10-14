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

from modal._utils.async_utils import synchronize_api
from modal.client import _Client
from modal_proto import api_pb2

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
    ids = set(known)
    ids.update(t.task_id for t in listed.tasks if t.app_description in owned)
    for task_id in ids:
        info = (
            await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=task_id))
        ).info
        if info.started_at and info.started_at < since:
            known.pop(task_id, None)
            continue
        known[task_id] = dict(
            task_id=task_id,
            started_at=info.started_at,
            finished_at=info.finished_at,
            gpu_type=info.gpu_type,
            gpu_count=info.gpu_config.count,
        )
    return known


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--clients", type=int, default=16)
    p.add_argument("--stagger-seconds", type=float, default=5.0)
    p.add_argument("--hours", type=float, default=0.5)
    p.add_argument("--environment", default="lilo-deploy")
    p.add_argument("--owned-apps", nargs="+", required=True)
    p.add_argument(
        "--resource-since",
        type=float,
        default=0.0,
        help="Exclude containers started before this experiment timestamp",
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
    error = None

    def launch(index):
        output = args.root / f"client-{index:02d}"
        with (args.root / f"client-{index:02d}.log").open("w") as log:
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
        processes.append(child)
        state.setdefault("launches", []).append(
            dict(client=index, pid=child.pid, time=time.time())
        )

    try:
        launch(0)
        while True:
            now = time.time()
            if stop or (args.root / "STOP_NOW").exists():
                raise RuntimeError("Stop requested")
            if now - state["started"] > args.hours * 3600:
                raise TimeoutError("Experiment deadline")
            traces = [
                events(args.root / f"client-{i:02d}" / "events.jsonl")
                for i in range(len(processes))
            ]
            if anchor is None and any(r["event"] == "ready" for r in traces[0]):
                anchor = time.monotonic()
                state["stagger_started_at"] = now
            if (
                anchor is not None
                and len(processes) < args.clients
                and time.monotonic() >= anchor + len(processes) * args.stagger_seconds
            ):
                launch(len(processes))
            for i, child in enumerate(processes):
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"Client {i} exited {child.returncode}")
            state.update(
                state="running" if anchor else "initializing",
                heartbeat=now,
                updates=[
                    sum(r["event"] == "update_done" for r in trace) for trace in traces
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
            if len(processes) == args.clients and all(c.poll() == 0 for c in processes):
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
