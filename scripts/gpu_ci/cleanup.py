"""Expiry backstop, deployed before GPUs; also supports immediate exact-run cleanup."""

import asyncio
import os
import re
import time

import modal
from modal.client import _Client
from modal_proto import api_pb2

RUN_PATTERN = re.compile(r"spindle-ci-dapo-(\d{10})-([a-f0-9]{12})")
APP_PATTERN = re.compile(
    r"(?:spindle-(?:trainer|inference|lora)-)?(spindle-ci-dapo-\d{10}-[a-f0-9]{12})"
)


def owned_apps(run_name):
    if not RUN_PATTERN.fullmatch(run_name):
        raise ValueError("Not a GPU CI run name")
    return [
        run_name,
        *[f"spindle-{role}-{run_name}" for role in ("trainer", "inference", "lora")],
    ]


def should_stop(name, now, run_name=None):
    match = APP_PATTERN.fullmatch(name)
    if not match:
        return False
    if run_name is not None:
        return name in owned_apps(run_name)
    return int(RUN_PATTERN.fullmatch(match[1])[1]) <= now


async def stop_apps(environment, run_name=None):
    client = await _Client.from_env()
    response = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=environment)
    )
    stopped = {}
    errors = []
    # Stop ingress/provisioners first, then repeat on the next janitor tick if a
    # deployment already in flight finishes after this scan.
    for app in sorted(
        response.apps, key=lambda app: not RUN_PATTERN.fullmatch(app.description)
    ):
        if app.state == api_pb2.APP_STATE_STOPPED or not should_stop(
            app.description, time.time(), run_name
        ):
            continue
        try:
            await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app.app_id))
            stopped[app.description] = app.app_id
        except Exception as exc:
            errors.append(f"{app.description}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return stopped


environment = os.environ.get("MODAL_ENVIRONMENT", "spindle-ci")
app = modal.App("spindle-gpu-ci-janitor")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.5.5")
    .env({"MODAL_ENVIRONMENT": environment})
)


@app.function(image=image, schedule=modal.Period(minutes=10), timeout=300)
async def reap_expired():
    # Keep Modal's task context and client on the function's event loop.
    print(await asyncio.wait_for(stop_apps(environment), timeout=60))
