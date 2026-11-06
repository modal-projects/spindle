"""CPU rollout workers; proxy credentials stay inside Modal secrets."""

import asyncio
import gzip
import pickle
import re
import time

import modal
from transformers import AutoTokenizer

from spindle.inference.sampling import sample_task
from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)

from .config import config
from .trajectory import rollout

app = modal.App("spindle-mimo-mixed-rollouts")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        *CORE_PACKAGES,
        STITCH_PACKAGE,
        TINKER_PACKAGE,
        "transformers>=5.5.0",
        "pyarrow",
        "mimoagent[modal] @ git+https://github.com/XiaomiMiMo/mimoagent.git@c3df8469f250b15e7906487a6d52be4c1827a509",
    )
    .add_local_python_source("spindle", "mixed_samplers")
)


@app.function(
    image=image,
    cpu=2,
    memory=16384,
    timeout=18000,
    max_containers=512,
    secrets=[modal.Secret.from_name("spindle-proxy")],
)
def sample_trajectory(task, gateway, group, model_id, version, seed):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.model_revision
    )
    row = asyncio.run(rollout(task, gateway, group, model_id, version, seed, tokenizer))
    return gzip.compress(pickle.dumps(row, protocol=5), compresslevel=1)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=3600,
    secrets=[modal.Secret.from_name("spindle-proxy")],
)
async def measure_capacity(gateway, model_id, version):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.model_revision
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Write a Python merge sort implementation and explain it.",
            }
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )

    async def sample(index, input_tokens=prompt):
        return await sample_task(
            {
                "request_id": f"capacity-{index}",
                "model_id": model_id,
                "publish_version": version,
                "latest": True,
                "payload": {
                    "prompt": {
                        "chunks": [{"type": "encoded_text", "tokens": input_tokens}]
                    },
                    "sampling_params": {
                        "max_tokens": 256,
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "top_k": -1,
                    },
                    "return_sampling_probabilities": True,
                },
            },
            gateway,
            headers=proxy_auth_headers(),
        )

    arithmetic_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    arithmetic = await sample("arithmetic", arithmetic_tokens)
    answer = tokenizer.decode(arithmetic["sequences"][0]["tokens"])
    if not re.search(r"\b42\b", answer):
        raise RuntimeError(f"sampler failed arithmetic validation: {answer!r}")
    await sample("warmup")
    started = time.monotonic()
    results = await asyncio.gather(*(sample(i) for i in range(16)))
    tokens = sum(len(r["sequences"][0]["tokens"]) for r in results)
    return {
        "tokens_per_second": tokens / (time.monotonic() - started),
        "tokens": tokens,
    }
