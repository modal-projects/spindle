"""Run GLM-5.3 Flash LoRA RL on DAPO Math with BF16 rollouts.

PYTHONPATH=src modal run --detach scripts/run_glm53_dapo.py
Requires the BF16 model cached by tests/manual/prepare_glm53_bf16.py.
"""

import hashlib
import json
import os
import runpy
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import modal
import modal.experimental

from spindle.deployments import DeploymentConfig, config_path, load
from spindle.inference.serving import start_lora_sidecar, terminate, wait_http
from spindle.providers.modal.glm53_image import inference_image, trainer_image
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)
from spindle.providers.modal.ray_cluster import start_trainer_cluster

app = modal.App("spindle-glm53-dapo-bf16")
BULLETIN_VOLUME = "spindle-glm53-pr26-rl-bulletin"
RESULTS_VOLUME = "spindle-glm53-dapo"
assets = modal.Volume.from_name("spindle-glm53-pr26-full-validation")
bulletin = modal.Volume.from_name(BULLETIN_VOLUME, version=2)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)
volumes = {
    "/validation": assets,
    "/bulletin": bulletin,
    "/checkpoints": results,
    KERNEL_CACHE_ROOT: kernel_cache_volume,
}
cache_env = {**KERNEL_CACHE_ENV, "TILELANG_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/tilelang"}


@app.cls(
    image=inference_image,
    gpu=["H200:8", "B200:8"],
    cpu=32,
    memory=262144,
    timeout=7200,
    startup_timeout=3600,
    scaledown_window=3600,
    max_containers=1,
    experimental_options={"efa_enabled": True},
    env=cache_env,
    volumes=volumes,
)
@modal.concurrent(max_inputs=128)
@modal.experimental.clustered(1, rdma=True)
class Sampler:
    settings_json: str = modal.parameter()

    @modal.enter()
    def start(self):
        assets.reload()
        path = "/validation/model-bf16"
        assert Path(path, "conversion.json").exists(), "Prepare BF16 weights first"
        self.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spindle.inference.sglang",
                path,
                self.settings_json,
            ],
            start_new_session=True,
        )
        wait_http("http://127.0.0.1:8001/health", self.server, 3300)
        self.sidecar = start_lora_sidecar(
            port=8000,
            sglang_port=8001,
            bulletin_root="/bulletin",
            bulletin_volume=BULLETIN_VOLUME,
        )
        wait_http("http://127.0.0.1:8000/health", self.sidecar, 120)

    @modal.method()
    def request(self, operation: str, payload=None):
        if operation == "stop":
            self.stop()
            return
        with httpx.Client(timeout=6900) as client:
            if operation == "ready":
                response = client.get("http://127.0.0.1:8001/server_info")
            elif operation == "generate":
                response = client.post("http://127.0.0.1:8000/generate", json=payload)
            else:
                raise ValueError(f"Unknown sampler operation: {operation}")
            if response.is_error:
                raise RuntimeError(
                    f"Sampler HTTP {response.status_code}: {response.text}"
                )
            return response.json()

    @modal.exit()
    def stop(self):
        terminate(getattr(self, "sidecar", None))
        terminate(getattr(self, "server", None))


@app.function(
    image=trainer_image,
    gpu="H200:8",
    cpu=32,
    memory=262144,
    timeout=86400,
    region="us",
    experimental_options={"efa_enabled": True},
    env={
        **cache_env,
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "SPINDLE_CHECKPOINT_VOLUME": RESULTS_VOLUME,
        "SPINDLE_BULLETIN_VOLUME": BULLETIN_VOLUME,
        "SPINDLE_BULLETIN_ROOT": "/bulletin",
    },
    volumes=volumes,
)
@modal.experimental.clustered(4, rdma=True)
def train(programs: dict, settings: dict, sampler_settings: str, experiment: dict):
    address = start_trainer_cluster(
        4, before_head=assets.reload, before_worker_join=assets.reload
    )
    if address is None:
        return
    os.environ["SPINDLE_RAY_ADDRESS"] = address
    for name, source in programs.items():
        Path("/tmp", name).write_text(source)
    sys.path.insert(0, "/tmp")
    try:
        namespace = runpy.run_path("/tmp/glm53_dapo.py")
        return namespace["train"](
            settings, Sampler(settings_json=sampler_settings), experiment
        )
    finally:
        subprocess.run(
            ["ray", "stop", "--force"], stdout=subprocess.DEVNULL, check=False
        )


@app.function(image=trainer_image, cpu=4, memory=16384, timeout=86400, volumes=volumes)
def run(programs: dict, settings: dict, sampler_settings: str, experiment: dict):
    """Keep orchestration remote so disconnecting the launch terminal is harmless."""
    for name, source in programs.items():
        Path("/tmp", name).write_text(source)
    sys.path.insert(0, "/tmp")
    namespace = runpy.run_path("/tmp/glm53_dapo_data.py")
    assets.reload()
    results.reload()
    namespace["prepare"](experiment)
    if experiment["prepare_only"]:
        return {"run_id": experiment["run_id"], "prepared": True}
    sampler = Sampler(settings_json=sampler_settings)
    sampler_ready = False
    try:
        info = sampler.request.remote("ready")
        assert info["dtype"] == "bfloat16" and info["quantization"] is None
        sampler_ready = True
        print("DAPO SAMPLER READY", flush=True)
        if not experiment["resume"]:
            namespace["baseline"](sampler, experiment)
        call = train.spawn(programs, settings, sampler_settings, experiment)
        print("DAPO TRAINER", call.object_id, flush=True)
        return call.get()
    except BaseException as exc:
        namespace["write_status"](experiment, "failed", error=repr(exc))
        raise
    finally:
        try:
            if sampler_ready:
                sampler.request.remote("stop")
        finally:
            sampler.update_autoscaler(min_containers=0, max_containers=0)


@app.local_entrypoint()
def main(
    steps: int = 30,
    clients: int = 4,
    groups: int = 32,
    group_size: int = 8,
    max_tokens: int = 8192,
    learning_rate: float = 1e-5,
    checkpoint_every: int = 5,
    eval_prompts: int = 64,
    eval_every: int = 5,
    concurrency: int = 128,
    prepare_only: bool = False,
    resume: str = "",
):
    if (
        min(
            steps,
            groups,
            group_size - 1,
            max_tokens,
            checkpoint_every,
            eval_prompts,
            eval_every,
        )
        < 1
    ):
        raise ValueError("Use positive run sizes, and at least two responses per group")
    if not 1 <= clients <= 4 or not 1 <= concurrency <= 128 or max_tokens > 8192:
        raise ValueError(
            "This recipe supports up to four clients, 128 concurrent requests and 8192 generated tokens"
        )
    config = DeploymentConfig.create(load(config_path("glm53-flash-lora-16k")))
    inference = dict(config.inference_settings)
    inference.update(
        quantization=None,
        dtype="bfloat16",
        disable_cuda_graph=True,
        max_running_requests=concurrency,
        max_queued_requests=concurrency,
        model_loader_extra_config=json.dumps(
            {"enable_multithread_load": True, "num_threads": 2}
        ),
    )
    experiment = {
        "run_id": "glm53-dapo-" + uuid.uuid4().hex[:12],
        "steps": steps,
        "clients": clients,
        "groups": groups,
        "group_size": group_size,
        "max_tokens": max_tokens,
        "learning_rate": learning_rate,
        "checkpoint_every": checkpoint_every,
        "eval_prompts": eval_prompts,
        "eval_every": eval_every,
        "concurrency": concurrency,
        "resume": resume,
        "prepare_only": prepare_only,
        "seed": 42,
        "dataset": "zhuzilin/dapo-math-17k",
        "dataset_revision": "2e65612930298bde4c5d58fd97b3f23a483aaff9",
        "results_volume": RESULTS_VOLUME,
        "context_length": 16384,
        "max_candidate_groups": groups * 8,
        "clip_low": 0.8,
        "clip_high": 1.28,
        "enable_thinking": True,
        "loss_normalization": "mean over non-truncated response tokens",
        "reward": "Miles math_dapo integer-answer correctness (+1/-1)",
    }
    programs = {
        name: Path(__file__).with_name(name).read_text()
        for name in ("glm53_dapo.py", "glm53_dapo_data.py")
    }
    experiment["source_sha256"] = {
        name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in programs.items()
    }
    call = run.spawn(
        programs, config.trainer_settings["miles"], json.dumps(inference), experiment
    )
    print(
        "DAPO LAUNCH",
        json.dumps({"app_id": app.app_id, "call_id": call.object_id, **experiment}),
        flush=True,
    )
    print("DAPO COMPLETE", call.get(), flush=True)
