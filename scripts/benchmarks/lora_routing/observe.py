"""Read adapter registration and request load from each benchmark replica."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import httpx
import modal
from modal._utils.async_utils import synchronize_api
from modal.client import _Client
from modal_proto import api_pb2

app = modal.App("spindle-lora-routing-observer")
image = modal.Image.debian_slim(python_version="3.12").pip_install("modal", "httpx")


@synchronize_api
async def replicas(app_name):
    client = await _Client.from_env()
    function = await client.stub.FunctionGet(
        api_pb2.FunctionGetRequest(
            app_name=app_name, object_tag="Server", environment_name="lilo-deploy"
        )
    )
    result = await client.stub.FlashContainerList(
        api_pb2.FlashContainerListRequest(function_id=function.function_id)
    )
    return list(result.containers)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=3600,
    region="us-west",
)
async def observe(app_names, seconds):
    deadline = time.monotonic() + seconds
    async with httpx.AsyncClient(
        headers={
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        },
        timeout=3,
    ) as client:
        while time.monotonic() < deadline:
            for app_name in app_names:
                row = {"time": time.time(), "app": app_name, "replicas": []}
                try:
                    gateway = await modal.Server.from_name(
                        app_name, "Server"
                    ).get_url.aio()
                    nodes = await replicas.aio(app_name)

                    async def inspect(node):
                        host = f"{node.host}:{node.port}" if node.port else node.host
                        try:
                            response = await client.get(
                                gateway + "/routing-state",
                                headers={"Modal-Flash-Upstream": host},
                            )
                            response.raise_for_status()
                            state = response.json()
                            assert state["replica"] == node.task_id
                            return state
                        except Exception as exc:
                            return {"replica": node.task_id, "error": str(exc)}

                    row["replicas"] = await asyncio.gather(
                        *(inspect(n) for n in nodes if n.host)
                    )
                except Exception as exc:
                    row["error"] = str(exc)
                yield row
            await asyncio.sleep(5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apps", nargs="+", required=True)
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with modal.enable_output(), app.run(), args.output.open("a") as out:
        for row in observe.remote_gen(args.apps, args.seconds):
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
