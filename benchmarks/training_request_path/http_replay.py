"""Three real localhost HTTP processes, real SDK, archived DAPO data, no GPU.

Use to isolate submission overhead, not to claim GPU training throughput.
Backend checks all received tensors and returns small synthetic training results.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import uvicorn

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine import Engine
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.http import HttpEngineClient, create_engine_app
from spindle.engine.operations import serialize_operation_payload
from spindle.providers.local import InMemoryKeyValueStore
from tests.support import SingleEnginePlatform, TinkerStubExecutor

ROOT = Path(__file__).resolve().parent


class AuditedStub(TinkerStubExecutor):
    async def execute_forward_backward_batch(self, executions):
        arrived = time.time()
        for command in executions:
            raw = serialize_operation_payload(command.payload)
            digest = hashlib.sha256(
                json.dumps(
                    raw, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            print(
                json.dumps(
                    {
                        "mark": "benchmark.backend_payload",
                        "model_id": command.model_id,
                        "sha256": digest,
                        "arrived": arrived,
                    }
                ),
                flush=True,
            )
        return await super().execute_forward_backward_batch(executions)


def worker(role, port, upstream):
    if role == "backend":
        app = create_backend_app(AuditedStub())
    elif role == "engine":
        app = create_engine_app(Engine(HttpBackendClient(upstream)))
    else:
        definition = SimpleNamespace(
            definition_id="replay",
            name="replay",
            model="Qwen/Qwen3.5-9B",
            parameterization="lora",
            max_context_length=32768,
        )
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            SingleEnginePlatform("replay", HttpEngineClient(upstream)),
        )
        app = create_control_plane_app(
            plane, (definition,), api_key="tml-local-benchmark", retrieve_window=1.0
        )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["backend", "engine", "frontend"])
    p.add_argument("--port", type=int)
    p.add_argument("--upstream")
    p.add_argument("--archive", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.role:
        worker(a.role, a.port, a.upstream)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    children = []
    logs = []
    url = None
    env = {
        **os.environ,
        "SPINDLE_REQUEST_TIMING": "1",
        "TINKER_API_KEY": "tml-local-benchmark",
    }
    try:
        for role in ["backend", "engine", "frontend"]:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            cmd = [sys.executable, "-u", __file__, "--role", role, "--port", str(port)]
            if url:
                cmd += ["--upstream", url]
            log = (a.output / f"{role}.log").open("w")
            logs.append(log)
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            url = f"http://127.0.0.1:{port}"
            health = {
                "backend": "/healthz",
                "engine": "/api/v1/models",
                "frontend": "/api/v1/healthz",
            }[role]
            deadline = time.monotonic() + 30
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"{role} exited; see its log")
                try:
                    if httpx.get(
                        url + health, headers={"X-API-Key": "tml-local-benchmark"}
                    ).is_success:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(role)
                time.sleep(0.1)
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "replay.py"),
                "--archive",
                str(a.archive),
                "--base-url",
                url,
                "--output",
                str(a.output / "clients"),
                "--updates",
                "3",
            ],
            env=env,
            check=True,
            timeout=1200,
        )
    finally:
        for child in reversed(children):
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
