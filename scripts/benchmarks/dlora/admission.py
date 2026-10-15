"""Observe engine admission during a fixed-work burst through Flash."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
import modal

from calibrate import app, image
from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.lora_routing import create_router


@app.function(
    image=image.add_local_python_source("calibrate"),
    cpu=2,
    timeout=300,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
)
async def diagnose(pool, source):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=36, slots=8)
    await router._refresh()
    replica = next(iter(router.replicas.values()))
    headers = proxy_auth_headers() | {"Modal-Flash-Upstream": replica.upstream}
    events = []
    retries = []
    began = time.monotonic()
    async with httpx.AsyncClient(headers=headers, timeout=120, http2=True) as client:
        async with httpx.AsyncClient(headers=headers, timeout=10) as observer:

            async def request(index):
                for attempt in range(1000):
                    response = await client.post(
                        gateway + "/generate",
                        json={
                            "input_ids": source["prompts"][0][:256],
                            "weight_run_id": source["model_id"],
                            "weight_version": {"exact_version": source["versions"][0]},
                            "sampling_params": {
                                "max_new_tokens": 256,
                                "ignore_eos": True,
                                "temperature": 0,
                            },
                        },
                    )
                    if response.status_code != 503:
                        response.raise_for_status()
                        return {
                            "index": index,
                            "seconds": time.monotonic() - began,
                            "retries": attempt,
                        }
                    retries.append(
                        {
                            "time": time.monotonic() - began,
                            "body": response.text[:200],
                            "capacity": response.headers.get(
                                "x-spindle-capacity-rejected"
                            ),
                        }
                    )
                    await asyncio.sleep(0.02)
                raise RuntimeError("Retries exhausted")

            tasks = [asyncio.create_task(request(i)) for i in range(36)]
            while not all(t.done() for t in tasks):
                response = await observer.get(gateway + "/benchmark-state")
                response.raise_for_status()
                state = response.json()
                events.append(
                    {
                        "time": time.monotonic() - began,
                        "active": state["active"],
                        "mode": state["mode"],
                        "loads": state["engine"]["loads"],
                    }
                )
                await asyncio.sleep(0.1)
            results = await asyncio.gather(*tasks)
    await router.aclose()
    return {
        "replica": replica.replica_id,
        "events": events,
        "results": results,
        "rejections": retries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads((args.root / "pools.json").read_text())[0]["pool"]
    source = json.loads(args.clients.read_text())[0]
    with modal.enable_output(), app.run(environment_name="lilo-deploy"):
        result = diagnose.remote(pool, source)
    (args.root / "admission.json").write_text(json.dumps(result, indent=2))
