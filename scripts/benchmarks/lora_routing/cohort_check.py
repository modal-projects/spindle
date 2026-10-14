"""Check a prepared hybrid-cache boundary against uncached greedy generation."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random

import httpx
import modal

from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.lora_routing import create_router
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "h2>=4,<5")
    .env({"MODAL_ENVIRONMENT": "lilo-deploy"})
    .add_local_python_source("spindle")
)

app = modal.App("spindle-cohort-cache-check")


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=300,
)
async def check(pool, source, http2=False):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=64, slots=8)
    await router._refresh()
    assert len(router.replicas) == 8
    vocabulary = sorted({t for p in source["prompts"] for t in p})
    prompt = random.Random(97731).choices(vocabulary, k=8192)
    async with httpx.AsyncClient(
        headers=proxy_auth_headers(), timeout=120, trust_env=False, http2=http2
    ) as client:

        async def one(replica):
            headers = {"Modal-Flash-Upstream": replica.upstream}

            async def flush():
                response = await client.post(
                    gateway + "/benchmark-flush-cache", headers=headers
                )
                response.raise_for_status()

            async def request(tokens, length):
                response = await client.post(
                    gateway + "/generate",
                    headers=headers,
                    json={
                        "input_ids": tokens,
                        "weight_run_id": source["model_id"],
                        "weight_version": {
                            "exact_version": source["versions"][0],
                            "min_version": None,
                        },
                        "sampling_params": {
                            "max_new_tokens": length,
                            "ignore_eos": True,
                            "temperature": 0,
                        },
                        "return_logprob": True,
                    },
                )
                response.raise_for_status()
                assert not http2 or response.http_version == "HTTP/2"
                result = response.json()
                assert (
                    result["meta_info"]["weight_version_start"]
                    == result["meta_info"]["weight_version_end"]
                    == source["versions"][0]
                )
                return result

            await flush()
            cold = await request(prompt, 16)
            await flush()
            seed = await request(prompt[:8128], 1)
            warm = await request(prompt, 16)
            assert cold["output_ids"] == warm["output_ids"], (
                "Cache preparation changed serialized greedy output"
            )
            assert warm["meta_info"]["cached_tokens"] >= 8064, warm["meta_info"]
            assert len(seed["output_ids"]) == 1
            return {
                "replica": replica.replica_id,
                "http2": http2,
                "cold": cold,
                "seed": seed,
                "warm": warm,
                "output_sha256": hashlib.sha256(
                    json.dumps(cold["output_ids"]).encode()
                ).hexdigest(),
            }

        results = await asyncio.gather(*(one(r) for r in router.replicas.values()))
    await router.aclose()
    assert len({r["output_sha256"] for r in results}) == 1
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--http2", action="store_true")
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        results = check.remote(
            json.loads(args.pools.read_text())[0]["pool"],
            json.loads(args.clients.read_text())[0],
            args.http2,
        )
    args.output.write_text(json.dumps(results, indent=2))
    print("Verified identical greedy output and boundary reuse on all eight replicas.")
