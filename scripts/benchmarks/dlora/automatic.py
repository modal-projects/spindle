"""Validate automatic specialization and exact-version handoff on a real engine."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import modal

from calibrate import app, image
from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.lora_routing import create_router


@app.function(
    image=image.add_local_python_source("calibrate"),
    cpu=2,
    timeout=900,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
)
async def validate(pool, sources, profile):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=36, slots=8)
    await router._refresh()
    replica = next(iter(router.replicas.values()))
    headers = proxy_auth_headers() | {"Modal-Flash-Upstream": replica.upstream}
    async with httpx.AsyncClient(headers=headers, timeout=600, http2=True) as client:
        response = await client.post(
            gateway + "/benchmark-mode", json=profile | {"enabled": True}
        )
        response.raise_for_status()

        async def request(source, output):
            response = await client.post(
                gateway + "/generate",
                json={
                    "input_ids": source["prompts"][0][:256],
                    "weight_run_id": source["model_id"],
                    "weight_version": {"exact_version": source["versions"][0]},
                    "sampling_params": {
                        "max_new_tokens": output,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            assert len(data["output_ids"]) == output
            assert data["meta_info"]["weight_version_start"] == source["versions"][0]
            return data["meta_info"]

        before = (await client.get(gateway + "/benchmark-state")).json()
        results = await asyncio.gather(*[request(sources[0], 1024) for _ in range(36)])
        specialized = (await client.get(gateway + "/benchmark-state")).json()
        assert (
            specialized["mode"]["merged_adapter"]
            == f"{sources[0]['model_id']}/weight_v{sources[0]['versions'][0]:06d}"
        ), specialized["mode"]
        assert any(r["lora_merged"] for r in results)
        changed = await request(sources[1], 8)
        after = (await client.get(gateway + "/benchmark-state")).json()
        assert after["mode"]["merged_adapter"] is None
        assert not changed["lora_merged"]
        response = await client.post(
            gateway + "/benchmark-mode", json=profile | {"enabled": False}
        )
        response.raise_for_status()
    await router.aclose()
    return {
        "before": before,
        "specialized": specialized,
        "after": after,
        "merged_requests": sum(r["lora_merged"] for r in results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--clients", required=True, type=Path)
    args = parser.parse_args()
    pool = json.loads((args.root / "pools.json").read_text())[0]["pool"]
    sources = json.loads(args.clients.read_text())
    profile = json.loads((args.root / "merge-profile.json").read_text())
    with modal.enable_output(), app.run(environment_name="lilo-deploy"):
        result = validate.remote(pool, sources, profile)
    (args.root / "automatic.json").write_text(json.dumps(result, indent=2))
