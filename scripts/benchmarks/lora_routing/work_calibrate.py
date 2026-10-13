"""Short real-GPU service calibration and serialized cache correctness control."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import httpx
import modal

from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.lora_routing import create_router
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)

app = modal.App("spindle-routing-calibration")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .env({"MODAL_ENVIRONMENT": "lilo-deploy"})
    .add_local_python_source("spindle")
)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=1800,
    nonpreemptible=True,
)
async def calibrate(pool, clients):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=64, slots=8)
    async with httpx.AsyncClient(
        headers=proxy_auth_headers(), timeout=900, trust_env=False
    ) as client:
        deadline = time.monotonic() + 1100
        while time.monotonic() < deadline:
            await router._refresh()
            print(f"ready: {len(router.replicas)}/8", flush=True)
            if len(router.replicas) == 8:
                break
            await asyncio.sleep(5)
        assert len(router.replicas) == 8

        async def one(replica):
            headers = {"Modal-Flash-Upstream": replica.upstream}
            rows = []

            async def request(source, tokens, output_tokens=1, label="", version=None):
                began = time.monotonic()
                version = source["versions"][0] if version is None else version
                response = await client.post(
                    gateway + "/generate",
                    headers=headers,
                    json=dict(
                        input_ids=tokens,
                        weight_run_id=source["model_id"],
                        weight_version=dict(exact_version=version, min_version=None),
                        sampling_params=dict(
                            max_new_tokens=output_tokens, ignore_eos=True, temperature=0
                        ),
                        return_logprob=True,
                    ),
                )
                response.raise_for_status()
                data = response.json()
                meta = data["meta_info"]
                assert (
                    meta["weight_version_start"]
                    == meta["weight_version_end"]
                    == version
                )
                row = dict(
                    label=label,
                    seconds=time.monotonic() - began,
                    meta=meta,
                    output_hash=hashlib.sha256(
                        json.dumps(data["output_ids"]).encode()
                    ).hexdigest(),
                )
                rows.append(row)
                return row

            source = clients[0]
            vocabulary = sorted({t for p in source["prompts"] for t in p})
            prompt = random.Random(7042).choices(vocabulary, k=8192)
            registration = await request(source, prompt[:128], label="registration")
            # Consecutive identical greedy requests test both native cache reuse
            # and output equivalence in a serialized, identical batch shape.
            controls = [
                await request(source, prompt, 16, "serialized-repeat") for _ in range(3)
            ]
            assert controls[-1]["meta"]["cached_tokens"] >= 7000, controls
            assert len({c["output_hash"] for c in controls}) == 1, controls
            state = (
                await client.get(gateway + "/routing-state", headers=headers)
            ).json()
            assert state["work"] is not None, state
            state_check = dict(
                cache_kind=state["work"]["cache_kind"],
                prefixes=len(state["work"]["prefixes"]),
                snapshot_seconds=state["work"]["snapshot_seconds"],
            )
            before = time.monotonic()
            prefills = await asyncio.gather(
                *(
                    request(
                        source,
                        random.Random(3000 + i).choices(vocabulary, k=8192),
                        label="uncached-prefill",
                    )
                    for i in range(8)
                )
            )
            prefill_wall = time.monotonic() - before
            prefill_rate = sum(
                8192 - r["meta"]["cached_tokens"] for r in prefills
            ) / max(r["meta"]["e2e_latency"] for r in prefills)
            await request(source, prompt, label="decode-prime")
            before = time.monotonic()
            decodes = await asyncio.gather(
                *(request(source, prompt, 512, "decode-36") for _ in range(36))
            )
            decode_wall = time.monotonic() - before
            decode_rate = 36 * 512 / max(r["meta"]["e2e_latency"] for r in decodes)
            # Reuse exactly the same short prompt/cache state for the cold GPU
            # adapter and hot GPU adapter timing pair. The CPU adapter stays loaded.
            for s in clients[:9]:
                await request(s, prompt[:128], label="adapter-prime")
                await request(s, prompt[:128], label="adapter-prime-repeat")
            transfers = []
            for repeat in range(3):
                for s in clients[1:9]:
                    await request(s, prompt[:128], label="adapter-displace")
                cold = await request(source, prompt[:128], label="gpu-cold")
                hot = await request(source, prompt[:128], label="gpu-hot")
                assert cold["meta"]["cached_tokens"] == hot["meta"]["cached_tokens"]
                transfers.append(
                    cold["meta"]["e2e_latency"] - hot["meta"]["e2e_latency"]
                )
            return dict(
                replica=replica.replica_id,
                rows=rows,
                prefill_rate=prefill_rate,
                decode_rate=decode_rate,
                prefill_wall=prefill_wall,
                decode_wall=decode_wall,
                gpu_transfer_differences=transfers,
                registration_exposed=registration["seconds"]
                - registration["meta"]["e2e_latency"],
                observation=state_check,
            )

        results = await asyncio.gather(*(one(r) for r in router.replicas.values()))
    await router.aclose()
    rates = dict(
        prefill_tokens_per_second=statistics.median(r["prefill_rate"] for r in results),
        decode_tokens_per_second=statistics.median(r["decode_rate"] for r in results),
        gpu_lora_seconds=max(
            0.001,
            statistics.median(
                t for r in results for t in r["gpu_transfer_differences"]
            ),
        ),
        registration_seconds=max(
            0.001, statistics.median(r["registration_exposed"] for r in results)
        ),
    )
    print(json.dumps(rates, indent=2), flush=True)
    return results, rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        results, rates = calibrate.remote(
            json.loads(args.pools.read_text())[0]["pool"],
            json.loads(args.clients.read_text()),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(results, indent=2))
    (args.output / "work-rates.json").write_text(json.dumps(rates, indent=2))
