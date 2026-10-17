"""Collect per-sequence completion records and replica counters every five seconds."""

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

app = modal.App("routing27-four-observer-1015")
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
    return function.handle_metadata._experimental_flash_urls[0], list(result.containers)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lilo-proxy", environment_name="lilo-deploy")],
    timeout=64800,
    nonpreemptible=True,
)
async def collect(pools, seconds):
    cursors = {}
    deadline = time.monotonic() + seconds
    async with httpx.AsyncClient(
        headers={
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        },
        timeout=4,
    ) as client:
        while time.monotonic() < deadline:

            async def pool_state(policy, pool):
                row = dict(time=time.time(), policy=policy, app=pool, replicas=[])
                try:
                    gateway, nodes = await asyncio.wait_for(
                        replicas.aio(pool), timeout=10
                    )

                    async def inspect(node):
                        upstream = (
                            f"{node.host}:{node.port}" if node.port else node.host
                        )
                        try:
                            response = await client.get(
                                gateway + "/benchmark-state",
                                params={"after": cursors.get(node.task_id, 0)},
                                headers={"Modal-Flash-Upstream": upstream},
                            )
                            response.raise_for_status()
                            state = response.json()
                            assert state["replica"] == node.task_id
                            for event in state["completions"]:
                                cursors[node.task_id] = max(
                                    cursors.get(node.task_id, 0), event["finished"]
                                )
                            state["request_count"] = len(state.pop("request_ids"))
                            work = state["work"]
                            if work is not None:
                                for key in ("prefixes", "request_ids", "prefix_nodes"):
                                    if key not in work:
                                        continue
                                    work[key + "_count"] = len(work.pop(key))
                            return state
                        except Exception as exc:
                            return dict(replica=node.task_id, error=str(exc))

                    row["replicas"] = await asyncio.gather(
                        *(inspect(n) for n in nodes if n.host)
                    )
                except Exception as exc:
                    row["error"] = str(exc)
                return row

            for row in await asyncio.gather(
                *(pool_state(policy, pool) for policy, pool in pools.items())
            ):
                yield row
            await asyncio.sleep(5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=28800)
    args = p.parse_args()
    pools = {
        policy: json.loads((args.root / policy / "owned-apps.json").read_text())[-1]
        for policy in ("default-no-split-k", "default", "stable", "model-spill")
    }
    with (
        modal.enable_output(),
        app.run(),
        (args.root / "observations.jsonl").open("a") as out,
    ):
        for row in collect.remote_gen(pools, args.seconds):
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
