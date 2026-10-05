"""Check the GLM image interfaces without downloading the 320B checkpoint.

PYTHONPATH=src modal run tests/manual/validate_glm53.py
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal.glm53_image import inference_image, trainer_image
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)

app = modal.App("spindle-glm53-validation")
artifacts = modal.Volume.from_name(
    "spindle-glm53-pr26-validation", create_if_missing=True
)


@app.function(
    image=trainer_image,
    gpu="H200",
    cpu=16,
    memory=131072,
    timeout=1800,
    env=KERNEL_CACHE_ENV,
    volumes={"/validation": artifacts, KERNEL_CACHE_ROOT: kernel_cache_volume},
)
def trainer(program: str = "", settings: dict | None = None):
    program = (
        program
        or """
import json
from transformers import AutoConfig
from megatron.bridge import AutoBridge
from spindle.backends.miles_runtime.runtime import MilesRuntime
from spindle.backends.miles_runtime.actor import SpindleMilesTrainRayActor
config = AutoConfig.from_pretrained("zai-org/GLM-5.3-Flash")
bridge = AutoBridge.from_hf_config(config)
provider = bridge.to_megatron_provider(load_weights=False)
assert provider.num_layers == 45
assert len(provider.kda_layers) == 34
assert provider.num_moe_experts == 288
assert provider.qk_pos_emb_head_dim == 0
print(json.dumps({"layers": provider.num_layers, "kda_layers": len(provider.kda_layers), "experts": provider.num_moe_experts}))
"""
    )
    script = Path("/tmp/glm53-validation.py")
    script.write_text(program)
    settings_path = Path("/tmp/glm53-settings.json")
    settings_path.write_text(json.dumps(settings))
    subprocess.run([sys.executable, "-u", str(script), str(settings_path)], check=True)
    return "Trainer check passed"


@app.function(
    image=inference_image,
    gpu="H200",
    cpu=16,
    memory=131072,
    timeout=1800,
    env=KERNEL_CACHE_ENV,
    volumes={"/validation": artifacts, KERNEL_CACHE_ROOT: kernel_cache_volume},
)
def inference(settings: dict, program: str = ""):
    program = (
        program
        or """
import json, sys
from sglang.srt.server_args import ServerArgs
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.lora.lora_manager import LoRAManager
from spindle.inference.sglang import main
import asyncio
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry

async def check_registry():
    registry = LoRARegistry()
    ref = LoRARef(lora_name="test", lora_path="/adapter", pinned=False)
    await registry.register(ref)
    counter = registry._counters[ref.lora_id]
    increment = counter.increment
    entered, resume = asyncio.Event(), asyncio.Event()
    async def delayed_increment(**kwargs):
        entered.set()
        await resume.wait()
        await increment(**kwargs)
    counter.increment = delayed_increment
    acquire = asyncio.create_task(registry.acquire(["test", "test"]))
    await entered.wait()
    unregister = asyncio.create_task(registry.unregister("test"))
    await asyncio.sleep(0)
    assert not unregister.done(), "Eviction raced an unpinned acquisition"
    resume.set()
    ids = await acquire
    await unregister
    assert counter.value() == 2
    await registry.release(ids)
    await registry.wait_for_unload(ref.lora_id)
    print("PASS: atomic adapter acquisition and balanced release")

asyncio.run(asyncio.wait_for(check_registry(), timeout=10))
args = ServerArgs(model_path="zai-org/GLM-5.3-Flash", **json.loads(sys.argv[1]))
print(json.dumps({"model": args.model_path, "context": args.context_length, "lora": args.enable_lora, "tp": args.tp_size}))
"""
    )
    script = Path("/tmp/glm53-inference-validation.py")
    script.write_text(program)
    subprocess.run(
        [sys.executable, "-u", str(script), json.dumps(settings)], check=True
    )
    return "Inference check passed"


@app.local_entrypoint()
def main(train: bool = False, sample: bool = False):
    config = DeploymentConfig.create(load(config_path("glm53-flash-lora-16k")))
    if sample and not train:
        program = Path(__file__).with_name("glm53_sampling.py").read_text()
        print(inference.spawn(config.inference_settings, program).get(), flush=True)
        return
    program = Path(__file__).with_name("glm53_training.py").read_text() if train else ""
    calls = [
        trainer.spawn(program, config.trainer_settings["miles"]),
        inference.spawn(config.inference_settings),
    ]
    for call in calls:
        print(call.object_id, flush=True)
    for call in calls:
        print(call.get(), flush=True)

    if sample:
        program = Path(__file__).with_name("glm53_sampling.py").read_text()
        print(inference.spawn(config.inference_settings, program).get(), flush=True)
