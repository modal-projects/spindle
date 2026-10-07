"""Full GLM validation on four RDMA-connected nodes.

PYTHONPATH=src modal run --detach tests/manual/glm53_multinode.py
Downloads the checkpoint on CPU before allocating 32 H200s. Runs two short
updates and one 16K update across four adapters, then checks publication and
checkpoint restore. All cluster containers stop when the check finishes.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import modal
import modal.experimental
from huggingface_hub import snapshot_download

from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal.glm53_image import trainer_image
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)
from spindle.providers.modal.ray_cluster import start_trainer_cluster

app = modal.App("spindle-glm53-full-validation")
volume_name = "spindle-glm53-pr26-full-validation"
artifacts = modal.Volume.from_name(volume_name, create_if_missing=True, version=2)
slice_artifacts = modal.Volume.from_name("spindle-glm53-pr26-validation")
prepare_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=1.0,<2"
)


@app.function(
    image=prepare_image,
    serialized=True,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={"/validation": artifacts, "/slice": slice_artifacts},
)
def prepare():
    latest = json.loads(Path("/slice/latest.json").read_text())
    adapter = Path(latest["adapter_path"].replace("/validation/", "/slice/"))
    weights = adapter / "adapter_model.safetensors"
    with weights.open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    dtypes = sorted({v["dtype"] for k, v in header.items() if k != "__metadata__"})
    print(
        json.dumps({"slice_adapter_bytes": weights.stat().st_size, "dtypes": dtypes}),
        flush=True,
    )
    path = snapshot_download(
        "zai-org/GLM-5.3-Flash",
        local_dir="/validation/model",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        max_workers=4,
    )
    artifacts.commit()
    print("Full checkpoint cached", path, flush=True)


@app.function(
    image=trainer_image,
    gpu="H200:8",
    cpu=32,
    memory=262144,
    timeout=7200,
    region="us",
    experimental_options={"efa_enabled": True},
    env={
        **KERNEL_CACHE_ENV,
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "SPINDLE_CHECKPOINT_VOLUME": volume_name,
        "SPINDLE_CHECKPOINT_ROOT": "/validation",
    },
    volumes={"/validation": artifacts, KERNEL_CACHE_ROOT: kernel_cache_volume},
)
@modal.experimental.clustered(4, rdma=True)
def trainer(program: str, settings: dict):
    address = start_trainer_cluster(
        4, before_head=artifacts.reload, before_worker_join=artifacts.reload
    )
    if address is None:
        return
    script = Path("/tmp/glm53-full-training.py")
    script.write_text(program)
    settings_path = Path("/tmp/glm53-full-settings.json")
    settings_path.write_text(json.dumps(settings))
    try:
        subprocess.run(
            [sys.executable, "-u", str(script), str(settings_path)],
            check=True,
            env={**os.environ, "SPINDLE_RAY_ADDRESS": address},
        )
        return "Full-model multinode validation passed"
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)


@app.local_entrypoint()
def main():
    config = DeploymentConfig.create(load(config_path("glm53-flash-lora-16k")))
    download = prepare.spawn()
    print("Preparing full checkpoint", download.object_id, flush=True)
    download.get()
    program = Path(__file__).with_name("glm53_full_training.py").read_text()
    call = trainer.spawn(program, config.trainer_settings["miles"])
    print("Four-node trainer", call.object_id, flush=True)
    print(call.get(), flush=True)
