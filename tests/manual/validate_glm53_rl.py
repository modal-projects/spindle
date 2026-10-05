"""Two full-model BF16 RL updates through Spindle's backend and sampler sidecar.

Run prepare_glm53_bf16.py first, then:
PYTHONPATH=src modal run --detach tests/manual/validate_glm53_rl.py
This bypasses the SDK frontend. It exercises real generation, rewards, the Miles
command backend, adapter publication, and loading the updated policy for sampling.
"""

import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import httpx
import modal
import modal.experimental
from fastapi.testclient import TestClient
from stitch.types import VersionRef
from transformers import AutoTokenizer

from spindle.deployments import DeploymentConfig, config_path, load
from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app
from spindle.inference.serving import start_lora_sidecar, terminate, wait_http
from spindle.providers.modal.glm53_image import inference_image, trainer_image
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)
from spindle.providers.modal.ray_cluster import start_trainer_cluster

app = modal.App("spindle-glm53-bf16-rl")
# Mounted Modal handles may be hydrated without their original names.
BULLETIN_VOLUME = "spindle-glm53-pr26-rl-bulletin"
CHECKPOINT_VOLUME = "spindle-glm53-pr26-rl-checkpoints"
assets = modal.Volume.from_name("spindle-glm53-pr26-full-validation")
bulletin = modal.Volume.from_name(BULLETIN_VOLUME, create_if_missing=True, version=2)
checkpoints = modal.Volume.from_name(
    CHECKPOINT_VOLUME, create_if_missing=True, version=2
)
cache_env = {**KERNEL_CACHE_ENV, "TILELANG_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/tilelang"}
volumes = {
    "/validation": assets,
    "/bulletin": bulletin,
    "/checkpoints": checkpoints,
    KERNEL_CACHE_ROOT: kernel_cache_volume,
}


@app.cls(
    image=inference_image,
    gpu=["H200:8", "B200:8"],
    cpu=32,
    memory=262144,
    timeout=3600,
    scaledown_window=1800,
    max_containers=1,
    region="us",
    env=cache_env,
    volumes=volumes,
)
@modal.concurrent(max_inputs=16)
class Sampler:
    settings_json: str = modal.parameter()

    @modal.enter()
    def start(self):
        assets.reload()
        path = "/validation/model-bf16"
        assert Path(path, "conversion.json").exists(), "Prepare BF16 weights first"
        settings = json.loads(self.settings_json)
        settings.update(
            quantization=None,
            dtype="bfloat16",
            disable_cuda_graph=True,
            max_running_requests=8,
            # Bound checkpoint prefetching: the default eight threads used ~1 TB.
            model_loader_extra_config=json.dumps(
                {"enable_multithread_load": True, "num_threads": 2}
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spindle.inference.sglang",
                path,
                json.dumps(settings),
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
    def ready(self):
        with httpx.Client(timeout=30) as client:
            response = client.get("http://127.0.0.1:8001/server_info")
            response.raise_for_status()
            return response.json()

    @modal.method()
    def check_exported_adapter(self, tokens):
        report = json.loads(Path("/validation/latest.json").read_text())
        with httpx.Client(base_url="http://127.0.0.1:8001", timeout=3000) as client:
            started = time.monotonic()
            response = client.post(
                "/load_lora_adapter",
                json={"lora_name": "preflight", "lora_path": report["adapter_path"]},
            )
            response.raise_for_status()
            registration_s = time.monotonic() - started
            try:
                started = time.monotonic()
                response = client.post(
                    "/generate",
                    json={
                        "input_ids": tokens,
                        "lora_path": "preflight",
                        "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                    },
                )
                response.raise_for_status()
                return {
                    "registration_s": registration_s,
                    "generation_s": time.monotonic() - started,
                    "sample": response.json(),
                }
            finally:
                response = client.post(
                    "/unload_lora_adapter", json={"lora_name": "preflight"}
                )
                response.raise_for_status()

    @modal.method()
    def encode(self, text: str):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @modal.method()
    def generate(self, payload: dict):
        with httpx.Client(timeout=3000) as client:
            response = client.post("http://127.0.0.1:8000/generate", json=payload)
            if response.is_error:
                raise RuntimeError(
                    f"Sampling HTTP {response.status_code}: {response.text}"
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
    timeout=7200,
    region="us",
    experimental_options={"efa_enabled": True},
    env={
        **cache_env,
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "SPINDLE_CHECKPOINT_VOLUME": CHECKPOINT_VOLUME,
        "SPINDLE_BULLETIN_VOLUME": BULLETIN_VOLUME,
        "SPINDLE_BULLETIN_ROOT": "/bulletin",
    },
    volumes=volumes,
)
@modal.experimental.clustered(4, rdma=True)
def train(program: str, settings: dict, sampler_settings: str):
    address = start_trainer_cluster(
        4, before_head=assets.reload, before_worker_join=assets.reload
    )
    if address is None:
        return
    os.environ["SPINDLE_RAY_ADDRESS"] = address
    path = Path("/tmp/glm53-rl.py")
    path.write_text(program)
    try:
        namespace = runpy.run_path(str(path))
        return namespace["main"](settings, Sampler(settings_json=sampler_settings))
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)


@app.function(image=trainer_image, timeout=3600)
def check_sampler(sampler_settings: str):
    # Resolve the class in the remote app instead of serializing a bound instance.
    return Sampler(settings_json=sampler_settings).ready.remote()


@app.function(image=inference_image, cpu=4, memory=16384, timeout=600, volumes=volumes)
def check_publication(run_id: str):
    def upstream(request):
        return httpx.Response(200, json={"meta_info": {}})

    reader = SnapshotBulletin(
        "/bulletin", refresh=modal.Volume.from_name(BULLETIN_VOLUME, version=2).reload
    )
    ref = VersionRef(run_id, 1)
    print(
        "SNAPSHOT", str(reader.snapshot_dir(ref)), "volume", BULLETIN_VOLUME, flush=True
    )
    with TestClient(
        create_app(reader, "http://upstream", transport=httpx.MockTransport(upstream))
    ) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": run_id,
                "weight_version": {"exact_version": 1},
            },
        )
        print("PUBLICATION RESPONSE", response.status_code, response.text, flush=True)
        response.raise_for_status()
    return "Cross-container snapshot resolution passed"


@app.local_entrypoint()
def main(publication_only: str = ""):
    if publication_only:
        print(check_publication.remote(publication_only))
        return
    config = DeploymentConfig.create(load(config_path("glm53-flash-lora-16k")))
    sampler = Sampler(settings_json=json.dumps(config.inference_settings))
    ready = sampler.ready.spawn()
    print("SAMPLER CALL", ready.object_id, sampler.ready.object_id, flush=True)
    info = ready.get()
    assert (
        check_sampler.remote(json.dumps(config.inference_settings))["dtype"]
        == "bfloat16"
    )
    assert info["quantization"] is None
    assert info["dtype"] == "bfloat16"
    print(
        "SAMPLER READY",
        json.dumps(
            {
                key: info[key]
                for key in (
                    "model_path",
                    "dtype",
                    "quantization",
                    "tp_size",
                    "ep_size",
                    "context_length",
                )
            }
        ),
        flush=True,
    )
    tokens = sampler.encode.remote("What is 7 times 8? Reply with just the number.")
    print(
        "BASE SAMPLE",
        sampler.generate.remote(
            {
                "input_ids": tokens,
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            }
        ),
        flush=True,
    )
    print(
        "EXPORTED ADAPTER CHECK",
        sampler.check_exported_adapter.remote(tokens),
        flush=True,
    )
    call = train.spawn(
        Path(__file__).with_name("glm53_rl.py").read_text(),
        config.trainer_settings["miles"],
        json.dumps(config.inference_settings),
    )
    print("RL TRAINER", call.object_id, flush=True)
    print("RL COMPLETE", call.get(), flush=True)
