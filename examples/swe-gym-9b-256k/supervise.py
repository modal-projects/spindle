"""Keep the eight SWE-Gym clients running until each finishes its batches.

On a later attempt this stops the owned apps, redeploys them, clears the
deployment failure, and starts the worker again. Each client resumes from
``eight-job{n}-checkpoint.json``.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from swe_gym_lora_benchmark import atomic_json

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run"
SRC = ROOT.parents[1] / "src"

stop = False


def on_stop(*_args):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_stop)
signal.signal(signal.SIGINT, on_stop)


def progress(target_jobs=8):
    counts = {}
    latest = {}
    for job in range(target_jobs):
        path = RUN / f"eight-job{job}-checkpoint.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        counts[job] = record.get("step", -1) + 1
        latest[job] = record.get("time", 0)
    return counts, latest


def modal_command(*args, environment=None):
    command = [sys.executable, "-m", "modal", *args]
    if environment:
        command.extend(["--env", environment])
    return command


def spindle_env():
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not current else str(SRC) + os.pathsep + current
    return env


def cleanup(manifest):
    environment = manifest.get("environment")
    results = {}
    for name in manifest["owned_apps"]:
        try:
            result = subprocess.run(
                modal_command("app", "stop", name, "--yes", environment=environment),
                capture_output=True,
                text=True,
                timeout=120,
            )
            results[name] = {
                "exit_code": result.returncode,
                "output": (result.stdout + result.stderr)[-1000:],
            }
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    atomic_json(RUN / "cleanup.json", {"time": time.time(), "apps": results})


def finished(counts, steps):
    return min(counts.values()) >= steps


def main():
    manifest = json.loads((RUN / "launch.json").read_text())
    steps = manifest["steps_per_job"]
    state = {"state": "starting", "pid": os.getpid(), "started": time.time()}
    failures = 0
    attempt = 0
    while (
        time.time() < manifest["absolute_deadline"]
        and not stop
        and not (RUN / "STOP_NOW").exists()
    ):
        before, _latest = progress()
        if finished(before, steps):
            state["state"] = "completed"
            break
        attempt += 1
        state.pop("error", None)
        state.update(attempt=attempt, state="starting", heartbeat=time.time())
        (RUN / "RESTART_REQUIRED.json").unlink(missing_ok=True)
        if attempt > 1:
            env = spindle_env()
            with (RUN / f"deploy-recovery-{attempt}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, str(ROOT / "deploy.py")],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            retry = [
                sys.executable,
                "-m",
                "spindle.deployment_cli",
                "deployment",
                "retry",
                "--frontend",
                manifest["frontend"],
            ]
            if manifest.get("environment"):
                retry.extend(["--env", manifest["environment"]])
            retry.append(manifest["definition_id"])
            subprocess.run(retry, env=env, check=True, timeout=120)
        with (RUN / f"worker-{attempt}.log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "train.py")],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state.update(state="running", worker_pid=child.pid)
        started = time.time()
        last_logs = started - 1
        last_watch = 0
        fatal = False
        events = RUN / "events.jsonl"
        offset = events.stat().st_size if events.exists() else 0
        starts = {}
        activity = {}
        updates = {}
        backend_failures = {}
        try:
            while child.poll() is None:
                now = time.time()
                counts, _latest = progress()
                state.update(heartbeat=now, completed_batches=counts)
                if stop or (RUN / "STOP_NOW").exists():
                    raise RuntimeError("stop requested")
                if now > manifest["absolute_deadline"]:
                    raise RuntimeError("wall-clock deadline")
                restart = RUN / "RESTART_REQUIRED.json"
                if restart.exists():
                    failure = restart.read_text()
                    fatal = any(
                        token in failure.lower()
                        for token in ("out of memory", "outofmemory", "nonfinite")
                    )
                    raise RuntimeError("worker requested recovery: " + failure)
                if shutil_free() < 20 * 1024**3:
                    raise RuntimeError("less than 20 GiB local disk remains")
                if events.exists():
                    with events.open() as stream:
                        stream.seek(offset)
                        while True:
                            line = stream.readline()
                            if not line:
                                break
                            try:
                                row = json.loads(line)
                            except ValueError:
                                break
                            offset = stream.tell()
                            job = row.get("job")
                            if row.get("phase") == "eight" and job is not None:
                                activity[job] = row["time"]
                                if row["event"] == "client_ready":
                                    starts[job] = row["time"]
                                    updates[job] = row["time"]
                                if row["event"] in ("optimizer_step", "step_done"):
                                    updates[job] = row["time"]
                if not starts and now - started > 3600:
                    raise TimeoutError("No client started within one hour")
                counts, _latest = progress()
                if finished(counts, steps):
                    state["state"] = "completed"
                    break
                for job, started_at in starts.items():
                    if counts[job] >= steps:
                        continue
                    if now - activity.get(job, started_at) > 1200:
                        raise TimeoutError(f"Client {job}: no events for 20 minutes")
                    if now - updates.get(job, started_at) > 10800:
                        raise TimeoutError(
                            f"Client {job}: no optimizer/checkpoint progress for three hours"
                        )
                if now - last_watch > 60:
                    last_watch = now
                    record_host(now, child.pid)
                    for app in manifest["monitored_apps"]:
                        since = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_logs)
                        )
                        logs = subprocess.run(
                            modal_command(
                                "app",
                                "logs",
                                app,
                                "--since",
                                since,
                                "--tail",
                                "1500",
                                environment=manifest.get("environment"),
                            ),
                            capture_output=True,
                            text=True,
                            timeout=40,
                        )
                        if logs.returncode:
                            backend_failures[app] = backend_failures.get(app, 0) + 1
                        else:
                            backend_failures[app] = 0
                        log_name = (
                            "trainer.log"
                            if app == manifest["trainer_app"]
                            else "sampler.log"
                        )
                        with (RUN / log_name).open("a") as handle:
                            handle.write(logs.stdout + logs.stderr)
                        fatal_markers = (
                            "OutOfMemoryError",
                            "CUDA out of memory",
                            "InductorError",
                            "RayTaskError",
                            "backend exited with code",
                        )
                        if any(marker in logs.stdout for marker in fatal_markers):
                            fatal = True
                            raise RuntimeError(
                                "Fatal GPU/backend error; inspect saved logs"
                            )
                    last_logs = now
                    if any(count >= 5 for count in backend_failures.values()):
                        raise RuntimeError("GPU log monitoring unavailable repeatedly")
                atomic_json(RUN / "supervisor.json", state)
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"Worker exited {child.returncode}")
        except BaseException as exc:
            state.update(error=f"{type(exc).__name__}: {exc}", state="failed")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        after, _latest = progress()
        state["completed_batches"] = after
        if finished(after, steps):
            state["state"] = "completed"
            break
        failures = 0 if sum(after.values()) > sum(before.values()) else failures + 1
        state.update(consecutive_failures=failures, heartbeat=time.time())
        atomic_json(RUN / "supervisor.json", state)
        cleanup(manifest)
        if fatal or failures >= 3 or stop or (RUN / "STOP_NOW").exists():
            break
        time.sleep(30)
    state.update(finished=time.time(), heartbeat=time.time())
    atomic_json(RUN / "supervisor.json", state)


def shutil_free():
    return shutil.disk_usage(ROOT).free


def record_host(now, pid):
    if psutil is None:
        return
    host = {
        "time": now,
        "cpu_percent": psutil.cpu_percent(),
        "available_memory": psutil.virtual_memory().available,
        "worker_rss": psutil.Process(pid).memory_info().rss,
    }
    with (RUN / "host-resources.jsonl").open("a") as handle:
        handle.write(json.dumps(host) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        path = RUN / "supervisor.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished=time.time(),
            heartbeat=time.time(),
        )
        atomic_json(path, state)
        raise
    finally:
        launch = RUN / "launch.json"
        if launch.exists():
            cleanup(json.loads(launch.read_text()))
