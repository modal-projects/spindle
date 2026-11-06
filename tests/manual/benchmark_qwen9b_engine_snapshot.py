"""Cold-start a production Spindle engine API over a snapshotted GPU backend.

Deploy first. Uses an isolated app/assets/cache; never registers with production.
The empty template has the real six-slot 16K recipe, warmed with a 2K step.
"""

import asyncio
import contextlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import modal

APP_NAME = os.environ.get(
    "SPINDLE_ENGINE_SNAPSHOT_APP", "spindle-qwen9b-engine-snapshot-1106"
)
MODEL = "Qwen/Qwen3.5-9B-Base"
TOKENS = int(os.environ.get("SPINDLE_ENGINE_SNAPSHOT_TOKENS", "2048"))
ASSET_VOLUME = "spindle-qwen9b-runtime-snapshot-assets-1105"
assets = modal.Volume.from_name(ASSET_VOLUME)
cache = modal.Volume.from_name(
    APP_NAME + "-kernel-cache", create_if_missing=True, version=2
)
app = modal.App(APP_NAME)
if modal.is_local():
    from spindle.providers.modal.miles_image import image as trainer_image

    image = trainer_image
    for filename in ("qwen9b_snapshot_probe.py", "qwen9b_engine_snapshot_runtime.py"):
        image = image.add_local_file(
            Path(__file__).with_name(filename), "/root/" + filename
        )
else:
    image = modal.Image.debian_slim()

ENV = {
    "PYTHONPATH": "/root:/root/miles:/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_DEBUG": "INFO",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_ENABLE_MONITORING": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    "TRITON_CACHE_DIR": "/kernel-cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/kernel-cache/inductor",
    "TILELANG_CACHE_DIR": "/kernel-cache/tilelang",
}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def payload():
    from spindle.engine.operations import serialize_operation_payload
    from tinker import Datum, ModelInput, TensorData
    from tinker.types.forward_backward_input import ForwardBackwardInput

    tokens = [100 + i % 100 for i in range(TOKENS + 1)]
    datum = Datum(
        ModelInput.from_ints(tokens[:-1]),
        {
            "target_tokens": TensorData(data=tokens[1:], dtype="int64", shape=[TOKENS]),
            "weights": TensorData(data=[1.0] * TOKENS, dtype="float32", shape=[TOKENS]),
        },
    )
    return serialize_operation_payload(
        ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
    )


def spec():
    return {
        "base_model": MODEL,
        "parameterization": "lora",
        "lora_config": {"rank": 32},
    }


class TrainerEngine:
    def control(self, command):
        import httpx

        with httpx.Client(timeout=300) as client:
            response = client.post(f"{self.backend_url}/snapshot_control/{command}")
            response.raise_for_status()
            return response.json()

    def initialize_backend(self):
        import httpx
        from spindle.configs.qwen35_9b_lora_16k import Config
        from spindle.deployments import DeploymentConfig

        started = time.monotonic()
        self.creator_task_id = os.environ["MODAL_TASK_ID"]
        self.initialization_id = uuid.uuid4().hex
        self.phase_times = {}
        assets.reload()
        self.deployment = DeploymentConfig.create(Config())
        port = free_port()
        self.backend_url = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "SPINDLE_BACKEND_CONFIG": json.dumps(self.deployment.trainer_settings),
            "SPINDLE_BASE_MODEL": MODEL,
        }
        self.backend_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spindle.engine.backend_http",
                "qwen9b_engine_snapshot_runtime:build_executor",
                str(port),
            ],
            env=env,
            start_new_session=True,
        )
        with httpx.Client(timeout=5) as client:
            deadline = time.monotonic() + 1500
            while time.monotonic() < deadline:
                if self.backend_process.poll() is not None:
                    raise RuntimeError(
                        f"backend exited: {self.backend_process.returncode}"
                    )
                try:
                    response = client.get(self.backend_url + "/healthz")
                    if response.is_success:
                        self.backend_health = response.json()
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("backend startup")
        self.phase_times["backend_initialized_s"] = time.monotonic() - started
        self.reference = asyncio.run(self.warm_backend())
        self.phase_times["backend_warmed_s"] = time.monotonic() - started
        self.before = self.control("probe")
        self.metadata = self.control("metadata")
        assert self.metadata["jobs"] == [], self.metadata
        self.initialization_s = time.monotonic() - started
        print(
            "ENGINE_SNAPSHOT_PHASE initialized",
            json.dumps(
                {
                    "id": self.initialization_id,
                    "creator": self.creator_task_id,
                    "times": self.phase_times,
                }
            ),
            flush=True,
        )

    async def warm_backend(self):
        from spindle.engine.api import Command, OperationKind
        from spindle.engine.backend_http import HttpBackendClient
        from spindle.engine.operations import parse_operation_payload
        from tinker import AdamParams

        client = HttpBackendClient(self.backend_url, read_timeout=600)
        model = "snapshot-warmup"
        try:
            await client.accept_model(model, spec())
            forward = parse_operation_payload(OperationKind.FORWARD, payload())
            reference = await client.execute(model, OperationKind.FORWARD, forward)
            await client.execute_forward_backward_batch(
                (Command(model, OperationKind.FORWARD_BACKWARD, forward),)
            )
            await client.execute(
                model, OperationKind.OPTIM_STEP, AdamParams(learning_rate=1e-4)
            )
            await client.unload_model(model)
            return reference
        finally:
            await client.close()

    def suspend_backend(self):
        self.suspended = self.control("suspend")
        print(
            "ENGINE_SNAPSHOT_PHASE ready_to_capture", self.initialization_id, flush=True
        )

    def recover_backend(self, *, suspended):
        import httpx
        import uvicorn
        from spindle.engine import Engine
        from spindle.engine.backend_http import HttpBackendClient
        from spindle.engine.http import create_engine_app

        self.restore_hook_at = time.time()
        start = time.monotonic()
        self.resumed = self.control("resume") if suspended else None
        self.resume_s = time.monotonic() - start
        self.engine_port = free_port()
        self.engine_token = secrets.token_urlsafe(24)
        self.frontend_id = uuid.uuid4().hex
        owner = self

        async def serve():
            executor = HttpBackendClient(owner.backend_url, read_timeout=600)
            owner.engine = Engine(executor, max_models=6, max_forward_backward_models=6)
            engine_app = create_engine_app(owner.engine, token=owner.engine_token)
            owner.server = uvicorn.Server(
                uvicorn.Config(
                    engine_app,
                    host="127.0.0.1",
                    port=owner.engine_port,
                    log_level="warning",
                )
            )
            await owner.server.serve()
            await owner.engine.close()

        self.frontend = threading.Thread(
            target=lambda: asyncio.run(serve()), daemon=True
        )
        self.frontend.start()
        self.engine_url = f"http://127.0.0.1:{self.engine_port}"
        with httpx.Client(
            headers={"Authorization": "Bearer " + self.engine_token}, timeout=5
        ) as client:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    response = client.get(self.engine_url + "/api/v1/models")
                    if response.is_success:
                        assert response.json()["model_ids"] == []
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.01)
            else:
                raise TimeoutError("engine HTTP readiness")
        self.ready_at = time.time()
        self.hook_to_ready_s = time.monotonic() - start
        print(
            "ENGINE_SNAPSHOT_PHASE engine_ready",
            json.dumps(
                {
                    "task": os.environ["MODAL_TASK_ID"],
                    "ready_at": self.ready_at,
                    "hook_to_ready_s": self.hook_to_ready_s,
                }
            ),
            flush=True,
        )

        # Functions can serve before asynchronous checkpoint publication ends.
        # Keep this short-lived producer alive like a real trainer server, so
        # single-use teardown does not race checkpoint upload/status publication.
        self.publication_hold_s = (
            120.0
            if suspended and os.environ["MODAL_TASK_ID"] == self.creator_task_id
            else 0.0
        )
        if self.publication_hold_s:
            print(
                "ENGINE_SNAPSHOT_PHASE publication_hold",
                self.publication_hold_s,
                flush=True,
            )
            time.sleep(self.publication_hold_s)

    @modal.method()
    def benchmark(self, client_sent_at: float):
        started = time.monotonic()
        after = self.control("probe")
        assert after["result"] == self.before["result"], (
            "restored model/optimizer/RNG/group fingerprints differ"
        )
        metadata = self.control("metadata")
        assert metadata == self.metadata, "backend process was rebuilt"
        training = asyncio.run(self.exercise_engine())
        row = {
            "task_id": os.environ["MODAL_TASK_ID"],
            "creator_task_id": self.creator_task_id,
            "initialization_id": self.initialization_id,
            "restored_from_different_container": os.environ["MODAL_TASK_ID"]
            != self.creator_task_id,
            "backend_metadata": metadata,
            "frontend_id": self.frontend_id,
            "initialization_s": self.initialization_s,
            "phase_times": self.phase_times,
            "restore_hook_at": self.restore_hook_at,
            "engine_ready_at": self.ready_at,
            "client_sent_at": client_sent_at,
            "client_to_engine_ready_wall_s": self.ready_at - client_sent_at,
            "resume_s": self.resume_s,
            "hook_to_engine_ready_s": self.hook_to_ready_s,
            "verification_s": time.monotonic() - started,
            "producer_publication_hold_s": self.publication_hold_s,
            "tokens": TOKENS,
            "environment": ENV,
            "before": self.before,
            "after": after,
            "training": training,
        }
        return row

    async def exercise_engine(self):
        import httpx
        from spindle.engine.http import HttpEngineClient
        from spindle.proto.responses import EncodedResult

        client = HttpEngineClient(self.engine_url, token=self.engine_token)
        client.http.timeout = httpx.Timeout(600)
        timings = {}

        async def finish(request_id):
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                state = await client.retrieve_future(request_id, timeout=20)
                assert state is not None
                if state.status.value != "pending":
                    assert state.status.value == "complete", state.error
                    return (
                        state.result.to_json()
                        if isinstance(state.result, EncodedResult)
                        else state.result
                    )
            raise TimeoutError(request_id)

        try:
            model = "fresh-client-" + uuid.uuid4().hex
            start = time.monotonic()
            assert await client.accept_model(model, spec())
            timings["accept_model_s"] = time.monotonic() - start
            start = time.monotonic()
            forward = await finish(
                await client.forward(
                    {"model_id": model, "seq_id": 1, "forward_input": payload()}
                )
            )
            timings["first_forward_s"] = time.monotonic() - start
            actual = forward["loss_fn_outputs"][0]["logprobs"]["data"]
            expected = self.reference["loss_fn_outputs"][0]["logprobs"]["data"]
            error = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
            assert error == 0, error
            start = time.monotonic()
            backward = await finish(
                await client.forward_backward(
                    json.dumps(
                        {
                            "model_id": model,
                            "seq_id": 2,
                            "forward_backward_input": payload(),
                        }
                    ).encode(),
                    "application/json",
                )
            )
            timings["first_forward_backward_s"] = time.monotonic() - start
            start = time.monotonic()
            optimizer = await finish(
                await client.optim_step(
                    {
                        "model_id": model,
                        "seq_id": 3,
                        "adam_params": {"learning_rate": 1e-4},
                    }
                )
            )
            timings["first_optimizer_s"] = time.monotonic() - start
            assert optimizer["metrics"]["update_successful:mean"] == 1, optimizer
            assert await client.model_ids() == (model,)
            await client.unload_model(model)
            assert await client.model_ids() == ()
            return {
                "timings": timings,
                "first_forward_max_error": error,
                "optimizer": optimizer,
                "backward_metrics": backward.get("metrics"),
                "fresh_model_accepted_trained_unloaded": True,
            }
        finally:
            await client.http.aclose()

    @modal.exit()
    def shutdown(self):
        if hasattr(self, "server"):
            self.server.should_exit = True
            self.frontend.join(timeout=10)
        if hasattr(self, "backend_process"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.backend_process.pid, signal.SIGTERM)


OPTIONS = dict(
    image=image,
    gpu="H100!:4",
    cpu=16,
    memory=65536,
    volumes={"/assets": assets, "/kernel-cache": cache},
    env=ENV,
    max_containers=1,
    single_use_containers=True,
    retries=0,
    timeout=1200,
    startup_timeout=2400,
)


@app.cls(**OPTIONS)
class Baseline(TrainerEngine):
    @modal.enter()
    def initialize(self):
        self.initialize_backend()
        self.recover_backend(suspended=False)


@app.cls(
    **OPTIONS,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class Snapshot(TrainerEngine):
    @modal.enter(snap=True)
    def initialize(self):
        self.initialize_backend()
        self.suspend_backend()

    @modal.enter(snap=False)
    def restore(self):
        self.recover_backend(suspended=True)


@app.local_entrypoint()
def main(
    mode: str = "snapshot",
    repeats: int = 3,
    output: str = "/tmp/qwen9b-engine-snapshot.json",
    require_restore: bool = True,
):
    if mode not in {"baseline", "snapshot"} or repeats < 1:
        raise ValueError("invalid mode/repeats")
    rows = []
    cls = modal.Cls.from_name(APP_NAME, mode.capitalize())
    for i in range(repeats):
        if i and mode == "snapshot":
            time.sleep(60)
        sent = time.time()
        start = time.monotonic()
        call = cls().benchmark.spawn(sent)
        print("ENGINE_SNAPSHOT_CALL", call.object_id, flush=True)
        try:
            row = call.get(timeout=3300)
        except BaseException as exc:
            with contextlib.suppress(Exception):
                call.cancel(terminate_containers=True)
            Path(output).write_text(
                json.dumps({"rows": rows, "failure": str(exc)}, indent=2)
            )
            raise
        row["client_to_result_s"] = time.monotonic() - start
        rows.append(row)
        Path(output).write_text(json.dumps({"mode": mode, "rows": rows}, indent=2))
        print(
            "ENGINE_SNAPSHOT_RESULT",
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"before", "after"}
                }
            ),
            flush=True,
        )
    if (
        mode == "snapshot"
        and require_restore
        and not any(row["restored_from_different_container"] for row in rows)
    ):
        raise RuntimeError("No separate-container restore observed")
