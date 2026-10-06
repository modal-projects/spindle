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
from contextlib import suppress
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
    replica_id: int = modal.parameter(default=0)

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
            terminate(getattr(self, "sidecar", None))
            terminate(getattr(self, "server", None))
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
            settings,
            [
                Sampler(settings_json=sampler_settings, replica_id=i)
                for i in range(experiment["rollout_replicas"])
            ],
            experiment,
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
    control_path = Path("/checkpoints") / experiment["run_id"] / "controller.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    reattach = control.get("app_id") == experiment["app_id"]
    if (
        experiment.get("continue_run")
        and not reattach
        and any(control_path.parent.glob("resume-*.json"))
    ):
        raise ValueError(
            "Optimizer checkpoints exist; use --resume instead of --continue-run"
        )
    namespace["prepare"](experiment)
    if experiment["prepare_only"]:
        return {"run_id": experiment["run_id"], "prepared": True}
    samplers = [
        Sampler(settings_json=sampler_settings, replica_id=i)
        for i in range(experiment["rollout_replicas"])
    ]
    sampler_ready = False
    trainer_call = None
    finished = False
    try:
        namespace["write_status"](
            experiment,
            "starting_samplers",
            rollout_replicas=len(samplers),
            rollout_gpus=8 * len(samplers),
        )
        ready_calls = [sampler.request.spawn("ready") for sampler in samplers]
        for call in ready_calls:
            info = call.get()
            assert info["dtype"] == "bfloat16" and info["quantization"] is None
        sampler_ready = True
        print("DAPO SAMPLER READY", flush=True)
        if reattach:
            trainer_call = modal.FunctionCall.from_id(control["trainer_call_id"])
            print("DAPO REATTACH", control["trainer_call_id"], flush=True)
        else:
            namespace["write_status"](experiment, "starting_trainer", trainer_gpus=32)
            trainer_call = train.spawn(programs, settings, sampler_settings, experiment)
            namespace["write_json"](
                control_path,
                {
                    "app_id": experiment["app_id"],
                    "trainer_call_id": trainer_call.object_id,
                },
            )
            results.commit()
            print("DAPO TRAINER", trainer_call.object_id, flush=True)
        outcome = trainer_call.get()
        finished = True
        return outcome
    except Exception as exc:
        finished = True
        if trainer_call is not None:
            with suppress(Exception):
                trainer_call.cancel(terminate_containers=True)
        namespace["write_status"](experiment, "failed", error=repr(exc))
        raise
    finally:
        # Preemption interrupts this CPU waiter with KeyboardInterrupt. Leave the
        # GPU call alive; the restarted waiter reattaches using controller.json.
        if finished:
            for sampler in samplers:
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
    concurrency: int = 16,
    rollout_replicas: int = 4,
    prepare_only: bool = False,
    validation_only: bool = False,
    stagger_s: float = 60.0,
    resume: str = "",
    continue_run: str = "",
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
    if (
        not 1 <= clients <= 4
        or not 1 <= concurrency <= 32
        or not 1 <= rollout_replicas <= 4
        or max_tokens > 8192
    ):
        raise ValueError(
            "This recipe supports four clients, four rollout replicas, 32 requests per replica and 8192 generated tokens"
        )
    if continue_run and (
        not continue_run.startswith("glm53-dapo-") or "/" in continue_run or resume
    ):
        raise ValueError(
            "Use a DAPO run ID for --continue-run, separately from --resume"
        )
    if resume or continue_run:
        raise ValueError(
            "Start a fresh run for independent clients and the new adapter targets"
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
        "run_id": continue_run or "glm53-dapo-" + uuid.uuid4().hex[:12],
        "continue_run": continue_run,
        "steps": steps,
        "clients": clients,
        "groups": groups,
        "group_size": group_size,
        "max_tokens": max_tokens,
        "learning_rate": learning_rate,
        "lora_rank": 16,
        "lora_alpha": 32,
        "beta1": 0.9,
        "beta2": 0.98,
        "weight_decay": 0.1,
        "routing_replay": True,
        "recipe": "glm53-kda-gates-r3-independent-v2",
        "checkpoint_every": checkpoint_every,
        "eval_prompts": eval_prompts,
        "eval_every": eval_every,
        "concurrency": concurrency * rollout_replicas,
        "rollout_replicas": rollout_replicas,
        "concurrency_per_replica": concurrency,
        "resume": resume,
        "prepare_only": prepare_only,
        "validation_only": validation_only,
        "stagger_s": stagger_s,
        "seed": 42,
        "dataset": "zhuzilin/dapo-math-17k",
        "dataset_revision": "2e65612930298bde4c5d58fd97b3f23a483aaff9",
        "results_volume": RESULTS_VOLUME,
        "context_length": 16384,
        "dynamic_sampling": False,
        "app_id": app.app_id,
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
