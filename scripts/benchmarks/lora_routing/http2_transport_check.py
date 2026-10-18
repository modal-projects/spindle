"""Inspect negotiated HTTP/2 stream limits and require a drained inference pool."""

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import time

import httpx
import modal

from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)
from spindle.providers.modal.lora_routing import create_router


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "h2>=4,<5")
    .env({"MODAL_ENVIRONMENT": "lilo-deploy"})
    .add_local_python_source("spindle")
)
app = modal.App("spindle-http2-transport-check")


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=240,
)
async def check(pool, allow_active=False):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=64, slots=8)
    await router._refresh()
    assert len(router.replicas) == 8
    async with httpx.AsyncClient(
        headers=proxy_auth_headers(), http2=True, timeout=20, trust_env=False
    ) as client:

        async def one(replica):
            response = await client.get(
                gateway + "/benchmark-state",
                headers={"Modal-Flash-Upstream": replica.upstream},
            )
            response.raise_for_status()
            assert response.http_version == "HTTP/2"
            return response.json()

        deadline = time.monotonic() + 120
        while True:
            states = await asyncio.gather(*(one(r) for r in router.replicas.values()))
            drained = all(
                s["active"] == 0
                and all(
                    v["num_running_reqs"] == v["num_waiting_reqs"] == 0
                    for v in s["engine"]["loads"]
                )
                for s in states
            )
            if drained or allow_active:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Canceled workload has not drained")
            await asyncio.sleep(5)
        connections = []
        for connection in client._transport._pool.connections:
            core = connection._connection
            connections.append(
                dict(
                    local_stream_limit=core._h2_state.local_settings.max_concurrent_streams,
                    remote_stream_limit=core._h2_state.remote_settings.max_concurrent_streams,
                    effective_stream_limit=core._max_streams,
                )
            )
    await router.aclose()
    return dict(
        at=time.time(),
        drained=drained,
        replicas=[s["replica"] for s in states],
        pool_active=sum(s["active"] for s in states),
        pool_running=sum(
            v["num_running_reqs"] for s in states for v in s["engine"]["loads"]
        ),
        pool_waiting=sum(
            v["num_waiting_reqs"] for s in states for v in s["engine"]["loads"]
        ),
        connections=connections,
        versions={
            name: importlib.metadata.version(name)
            for name in ("httpx", "httpcore", "h2")
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-active", action="store_true")
    args = parser.parse_args()
    pool = json.loads(args.pools.read_text())[0]["pool"]
    with modal.enable_output(), app.run():
        result = check.remote(pool, args.allow_active)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
