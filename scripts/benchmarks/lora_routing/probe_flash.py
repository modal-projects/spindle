"""CPU-only integration check for exact Flash routing and stale upstreams."""

import asyncio
import json
import os
import time
import threading
import uvicorn
from fastapi import FastAPI
import modal
import httpx
from modal._utils.async_utils import synchronize_api
from modal.client import _Client
from modal_proto import api_pb2

app = modal.App("spindle-routing-flash-probe")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi", "uvicorn", "httpx", "modal"
)


@app.server(
    image=image,
    min_containers=2,
    max_containers=2,
    port=8000,
    routing_region="us-west",
    compute_region="us-west",
)
class Server:
    @modal.enter()
    def start(self):
        web = FastAPI()

        @web.get("/")
        def identity():
            return {"replica": os.environ["MODAL_TASK_ID"]}

        threading.Thread(
            target=uvicorn.run,
            args=(web,),
            kwargs={"host": "0.0.0.0", "port": 8000},
            daemon=True,
        ).start()


@synchronize_api
async def containers(function_id):
    client = await _Client.from_env()
    return await client.stub.FlashContainerList(
        api_pb2.FlashContainerListRequest(function_id=function_id)
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=120,
    region="us-west",
)
async def probe(url, function_id):
    async with httpx.AsyncClient(
        headers={
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        },
        timeout=10,
    ) as http:
        deadline = time.monotonic() + 70
        while time.monotonic() < deadline:
            response = await containers.aio(function_id)
            if len(response.containers) >= 2:
                break
            await http.get(url)
            await asyncio.sleep(1)
        assert len(response.containers) == 2, "Probe needs two live replicas"
        verified = []
        for c in response.containers:
            host = f"{c.host}:{c.port}" if c.port else c.host

            async def request():
                r = await http.get(url, headers={"Modal-Flash-Upstream": host})
                return r.status_code, r.text

            samples = await asyncio.gather(*(request() for _ in range(20)))
            assert all(
                status == 200 and json.loads(body)["replica"] == c.task_id
                for status, body in samples
            ), samples
            verified.append(
                {"replica": c.task_id, "host": host, "matched": len(samples)}
            )
        r = await http.get(url, headers={"Modal-Flash-Upstream": "gone.invalid:443"})
        assert r.status_code == 503, r.text
        return {"verified": verified, "stale_status": r.status_code}


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        url = Server.get_url()
        print(json.dumps(probe.remote(url, Server.object_id), indent=2))
