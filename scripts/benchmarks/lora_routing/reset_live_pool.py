"""Remove prior-arm adapter and KV state from an idle measured inference pool."""

import argparse
import asyncio
import os
from pathlib import Path

import httpx
import modal

from observe import replicas
from supervise import write


app = modal.App("spindle-routing-live-reset")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal", "httpx")
    .add_local_python_source("observe", "supervise")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=300,
    nonpreemptible=True,
)
async def reset(pool):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    nodes = await replicas.aio(pool)
    if len(nodes) != 8 or any(not n.host for n in nodes):
        raise RuntimeError("Reset requires the original eight healthy replicas")
    async with httpx.AsyncClient(
        headers={
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        },
        timeout=120,
        trust_env=False,
    ) as client:

        async def one(node):
            host = f"{node.host}:{node.port}" if node.port else node.host
            headers = {"Modal-Flash-Upstream": host}

            async def inspect():
                response = await client.get(
                    gateway + "/benchmark-state", headers=headers
                )
                response.raise_for_status()
                state = response.json()
                assert state["replica"] == node.task_id
                assert state["active"] == 0
                assert all(
                    load["num_running_reqs"] == load["num_waiting_reqs"] == 0
                    for load in state["engine"]["loads"]
                )
                return state

            before = await inspect()
            response = await client.post(
                gateway + "/benchmark-unregister",
                headers=headers,
                json={"adapters": before["work"]["cpu_adapters"]},
            )
            response.raise_for_status()
            response = await client.post(
                gateway + "/benchmark-flush-cache", headers=headers
            )
            response.raise_for_status()
            # Wait for the engine's periodic state writer to publish the reset.
            for _ in range(20):
                after = await inspect()
                if not after["work"]["cpu_adapters"] and not after["work"]["prefixes"]:
                    return dict(replica=node.task_id, before=before, after=after)
                await asyncio.sleep(1)
            raise RuntimeError("Engine did not publish an empty adapter/cache state")

        return await asyncio.gather(*(one(n) for n in nodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing reset audit")
    with modal.enable_output(), app.run(environment_name="lilo-deploy"):
        result = reset.remote(args.pool)
    write(args.output, result)


if __name__ == "__main__":
    main()
