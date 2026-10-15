"""Check merged numerical agreement and exact restoration on fixed prompts."""

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
    timeout=1800,
    region="us-east",
    secrets=[modal.Secret.from_name("lilo-proxy")],
)
async def check(pool, sources, profile):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=36, slots=8)
    for _ in range(240):
        await router._refresh()
        if router.replicas:
            break
        await asyncio.sleep(5)
    assert router.replicas, "No healthy replica"
    replica = next(iter(router.replicas.values()))
    headers = proxy_auth_headers() | {"Modal-Flash-Upstream": replica.upstream}
    records = []
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        response = await client.post(
            gateway + "/benchmark-mode", json=profile | {"enabled": False}
        )
        response.raise_for_status()

        async def control(action, body):
            response = await client.post(
                gateway + "/benchmark-transition/" + action, json=body
            )
            response.raise_for_status()
            assert response.json()["success"]

        for source in sources[:2]:
            version = source["versions"][0]
            adapter = f"{source['model_id']}/weight_v{version:06d}"
            payload = dict(
                input_ids=source["prompts"][0][:256],
                weight_run_id=source["model_id"],
                weight_version={"exact_version": version},
                sampling_params=dict(max_new_tokens=32, ignore_eos=True, temperature=0),
                return_logprob=True,
                logprob_start_len=0,
            )

            async def generate():
                response = await client.post(gateway + "/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                assert (
                    data["meta_info"]["weight_version_start"]
                    == data["meta_info"]["weight_version_end"]
                    == version
                )
                assert len(data["output_ids"]) == 32
                return data

            base_response = await client.post(
                gateway + "/generate",
                json={
                    k: v
                    for k, v in payload.items()
                    if k not in {"weight_run_id", "weight_version"}
                },
            )
            base_response.raise_for_status()
            base_logprobs = base_response.json()["meta_info"]["input_token_logprobs"]
            before = await generate()
            await control("merge_lora_weights", {"lora_name": adapter})
            merged = await generate()
            await control("unmerge_lora_weights", {})
            after = await generate()
            a, b, c = [
                d["meta_info"]["input_token_logprobs"] for d in [before, merged, after]
            ]
            assert a == c and before["output_ids"] == after["output_ids"]
            assert len(a) == len(b)
            assert [v[1] for v in a] == [v[1] for v in b]
            differences = [abs(x[0] - y[0]) for x, y in zip(a, b) if x[0] is not None]
            mean, maximum = sum(differences) / len(differences), max(differences)
            # Same BF16 agreement bounds as the pinned PR's merge test.
            records.append(
                dict(
                    adapter=adapter,
                    input_ids=payload["input_ids"],
                    input_logprobs_unmerged=a,
                    input_logprobs_merged=b,
                    mean_logprob_delta=mean,
                    max_logprob_delta=maximum,
                    within_pr_tolerance=mean < 0.03 and maximum < 0.25,
                    unmerged_vs_base_mean=sum(
                        abs(x[0] - y[0])
                        for x, y in zip(a, base_logprobs)
                        if x[0] is not None
                    )
                    / len(differences),
                    merged_vs_base_mean=sum(
                        abs(x[0] - y[0])
                        for x, y in zip(b, base_logprobs)
                        if x[0] is not None
                    )
                    / len(differences),
                    exact_restore=True,
                    greedy_tokens_equal=before["output_ids"] == merged["output_ids"],
                )
            )
    await router.aclose()
    return dict(replica=replica.replica_id, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    with modal.enable_output(), app.run(environment_name="lilo-deploy"):
        result = check.remote(
            json.loads((root / "pools.json").read_text())[0]["pool"],
            json.loads((root / "clients16.json").read_text()),
            json.loads((root / "merge-profile.json").read_text()),
        )
    (root / "equivalence.json").write_text(json.dumps(result, indent=2))
    assert all(r["within_pr_tolerance"] for r in result["records"]), [
        (r["mean_logprob_delta"], r["max_logprob_delta"]) for r in result["records"]
    ]
