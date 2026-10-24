"""Bounded two-H200 RL validation using the production Modal Function runtime."""

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

import modal

from spindle.providers.modal.rollout_image import (
    SGLANG_REVISION,
    image as rollout_image,
)
from spindle.providers.modal.torchtitan_image import (
    TORCHTITAN_COMMIT,
    image as trainer_image,
)


def run_sampler(gateway_queue, environment):
    os.environ.update(environment)
    with modal.forward(8000) as tunnel:
        gateway_queue.put(tunnel.url)
        subprocess.run([sys.executable, "/root/torchtitan_rl_sampler.py"], check=True)


def run_trainer(environment):
    os.environ.update(environment)
    result = {"passed": False, "error": "worker did not emit a result; inspect logs"}
    process = subprocess.Popen(
        [sys.executable, "/root/torchtitan_rl_trainer.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("RL_RESULT="):
            result = json.loads(line.removeprefix("RL_RESULT="))
    process.wait()
    return result


def check_volume_access(environment):
    os.environ.update(environment)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, modal; "
            "v = modal.Volume.from_name(os.environ['SPINDLE_BULLETIN_VOLUME']); "
            "v.commit(); v.reload(); print('Volume commit/reload preflight passed')",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("use Python 3.12 to match the Modal Function images")
    if not 120 <= args.timeout_seconds <= 1800:
        parser.error("--timeout-seconds must be between 120 and 1800")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment_name = "lilo-deploy"
    volume_name = "spindle-titan-rl-" + uuid.uuid4().hex[:12]
    app = modal.App("spindle-torchtitan-rl-validation")
    environment = {
        "RL_MODEL": "Qwen/Qwen3.8-27B",
        "RL_MODEL_REVISION": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "SPINDLE_BASE_MODEL": "Qwen/Qwen3.8-27B",
        "SPINDLE_BULLETIN_ROOT": "/bulletin",
        "SPINDLE_BULLETIN_VOLUME": volume_name,
        "MODAL_ENVIRONMENT": environment_name,
        "RL_API_KEY": "tml-" + secrets.token_urlsafe(32),
        "RL_STEPS": str(args.steps),
        "PYTHONUNBUFFERED": "1",
    }
    modal.Volume.objects.create(
        volume_name, version=2, environment_name=environment_name
    )
    volume = modal.Volume.from_name(volume_name, environment_name=environment_name)
    resources = {
        "gpu": "H200",
        "cpu": 16,
        "memory": 131072,
        "timeout": args.timeout_seconds,
        "startup_timeout": min(120, args.timeout_seconds),
        "max_containers": 1,
        "retries": 0,
        "single_use_containers": True,
        "volumes": {"/bulletin": volume},
        "serialized": True,
    }
    sampler = app.function(
        image=rollout_image.add_local_file(
            Path(__file__).with_name("torchtitan_rl_sampler.py"),
            "/root/torchtitan_rl_sampler.py",
        ),
        **resources,
    )(run_sampler)
    trainer = app.function(
        image=trainer_image.add_local_file(
            Path(__file__).with_name("torchtitan_rl_trainer.py"),
            "/root/torchtitan_rl_trainer.py",
        ),
        **resources,
    )(run_trainer)
    preflight = app.function(
        image=trainer_image,
        timeout=120,
        retries=0,
        volumes={"/bulletin": volume},
        serialized=True,
        single_use_containers=True,
    )(check_volume_access)
    calls = []
    result = {"passed": False, "error": "validation did not complete; inspect logs"}
    started = time.time()
    try:
        with modal.enable_output(), app.run(environment_name=environment_name):
            preflight.remote(environment)
            with modal.Queue.ephemeral(environment_name=environment_name) as gateways:
                gpu_deadline = time.monotonic() + args.timeout_seconds
                sampler_call = sampler.spawn(gateways, environment)
                calls.append(sampler_call)
                environment["RL_SAMPLER_URL"] = gateways.get(timeout=180)
                trainer_call = trainer.spawn(environment)
                calls.append(trainer_call)
                print(
                    "RL_FUNCTION_CALLS="
                    + json.dumps([call.object_id for call in calls]),
                    flush=True,
                )
                try:
                    while True:
                        if time.monotonic() >= gpu_deadline:
                            raise TimeoutError("GPU experiment deadline exceeded")
                        try:
                            result = trainer_call.get(timeout=5)
                            break
                        except TimeoutError:
                            try:
                                sampler_call.get(timeout=0)
                            except TimeoutError:
                                continue
                            raise RuntimeError(
                                "sampler exited before training completed"
                            )
                finally:
                    for call in calls:
                        with suppress(Exception):
                            call.cancel(terminate_containers=True)
    finally:
        result.update(
            function_call_ids=[call.object_id for call in calls],
            elapsed_seconds=time.time() - started,
            torchtitan_commit=TORCHTITAN_COMMIT,
            sglang_revision=SGLANG_REVISION,
            gpu="2 x H200",
            timeout_seconds_per_gpu=args.timeout_seconds,
            model_revision=environment["RL_MODEL_REVISION"],
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        modal.Volume.objects.delete(volume_name, environment_name=environment_name)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
