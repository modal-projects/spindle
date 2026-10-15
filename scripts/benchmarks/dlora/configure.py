"""Switch only replica-local policy on an idle, unchanged physical pool."""

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
    cpu=1,
    timeout=1200,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
)
async def configure(pool, profile, replicas, sources=None):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=profile["capacity"], slots=8)
    for _ in range(180):
        await router._refresh()
        if len(router.replicas) == replicas:
            break
        await asyncio.sleep(5)
    assert len(router.replicas) == replicas
    async with httpx.AsyncClient(headers=proxy_auth_headers(), timeout=120) as client:

        async def one(replica):
            headers = {"Modal-Flash-Upstream": replica.upstream}
            response = await client.get(gateway + "/benchmark-config", headers=headers)
            response.raise_for_status()
            actual = response.json()
            assert actual["replica"] == replica.replica_id
            assert actual["engine"] == dict(
                max_running_requests=profile["capacity"],
                max_queued_requests=profile["capacity"] + profile["queue_buffer"],
                enable_lora_merge=True,
                lora_unmerge_mode="restore",
            ), actual
            response = await client.post(
                gateway + "/benchmark-mode",
                headers=headers,
                json=profile | {"enabled": False},
            )
            response.raise_for_status()
            if sources:
                state = (
                    await client.get(gateway + "/benchmark-state", headers=headers)
                ).json()
                response = await client.post(
                    gateway + "/benchmark-unregister",
                    headers=headers,
                    json={"adapters": state["work"]["cpu_adapters"]},
                )
                response.raise_for_status()
                for source in sources[:8]:
                    response = await client.post(
                        gateway + "/generate",
                        headers=headers,
                        json={
                            "input_ids": source["prompts"][0][:16],
                            "weight_run_id": source["model_id"],
                            "weight_version": {"exact_version": source["versions"][0]},
                            "sampling_params": {
                                "max_new_tokens": 1,
                                "ignore_eos": True,
                                "temperature": 0,
                            },
                        },
                    )
                    response.raise_for_status()
                adapter = f"{source['model_id']}/weight_v{source['versions'][0]:06d}"
                for action, payload in [
                    ("merge_lora_weights", {"lora_name": adapter}),
                    ("unmerge_lora_weights", {}),
                    ("flush_cache", {}),
                ]:
                    response = await client.post(
                        gateway + "/benchmark-transition/" + action,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    if action != "flush_cache":
                        assert response.json()["success"]
            response = await client.post(
                gateway + "/benchmark-mode", headers=headers, json=profile
            )
            response.raise_for_status()

        await asyncio.gather(*(one(r) for r in router.replicas.values()))
    result = dict(
        pool=pool, profile=profile, replicas=list(router.replicas), time=time.time()
    )
    await router.aclose()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--replicas", required=True, type=int)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warm-clients", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text()) | {"enabled": args.enabled}
    with modal.enable_output(), app.run(environment_name="lilo-deploy"):
        result = configure.remote(
            args.pool,
            profile,
            args.replicas,
            json.loads(args.warm_clients.read_text()) if args.warm_clients else None,
        )
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
