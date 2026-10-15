"""Real-engine correctness and paired merge calibration through Flash."""

import argparse
import asyncio
import json
from pathlib import Path
import random
import statistics
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

app = modal.App("spindle-dlora-calibration")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .add_local_python_source("spindle")
)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
)
async def calibrate(pool, sources):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=36, slots=8)
    async with httpx.AsyncClient(
        headers=proxy_auth_headers(),
        timeout=900,
        limits=httpx.Limits(max_connections=128),
        http2=True,
    ) as client:
        for _ in range(240):
            await router._refresh()
            if router.replicas:
                break
            await asyncio.sleep(3)
        assert router.replicas, "No healthy replica"
        replica = next(iter(router.replicas.values()))
        headers = {"Modal-Flash-Upstream": replica.upstream}
        source = sources[0]
        version = source["versions"][0]
        adapter = f"{source['model_id']}/weight_v{version:06d}"
        prompt = source["prompts"][0][:256]
        timings = []

        async def request(
            tokens, output=32, logprob=False, chosen=source, chosen_version=version
        ):
            for attempt in range(300):
                response = await client.post(
                    gateway + "/generate",
                    headers=headers,
                    json=dict(
                        input_ids=tokens,
                        weight_run_id=chosen["model_id"],
                        weight_version=dict(exact_version=chosen_version),
                        sampling_params=dict(
                            max_new_tokens=output, temperature=0, ignore_eos=True
                        ),
                        return_logprob=logprob,
                        logprob_start_len=0,
                    ),
                )
                if response.status_code != 503:
                    response.raise_for_status()
                    data = response.json()
                    assert len(data["output_ids"]) == output
                    assert data["meta_info"]["weight_version_start"] == chosen_version
                    return data
                await asyncio.sleep(0.02)
            raise RuntimeError("Admission retries exhausted")

        async def control(action, payload=None):
            began = time.monotonic()
            response = await client.post(
                gateway + "/benchmark-transition/" + action,
                headers=headers,
                json=payload or {},
            )
            response.raise_for_status()
            data = {} if action == "flush_cache" else response.json()
            if "success" in data:
                assert data["success"], data
            timings.append(dict(action=action, seconds=time.monotonic() - began))
            return data

        await request(prompt, 8)
        await control("flush_cache")
        before = await request(prompt, 32, True)
        # Register another exact version before specializing the replica.
        await request(
            sources[1]["prompts"][0][:128],
            8,
            chosen=sources[1],
            chosen_version=sources[1]["versions"][0],
        )
        await control("merge_lora_weights", {"lora_name": adapter})
        merged = await request(prompt, 32, True)
        await control("unmerge_lora_weights")
        after = await request(prompt, 32, True)
        assert before["output_ids"] == after["output_ids"], (
            "Unmerge changed greedy tokens"
        )
        assert (
            before["meta_info"]["input_token_logprobs"]
            == after["meta_info"]["input_token_logprobs"]
        ), "Unmerge failed exact logprob restoration"
        await request(
            sources[1]["prompts"][0][:128],
            8,
            chosen=sources[1],
            chosen_version=sources[1]["versions"][0],
        )
        state = (await client.get(gateway + "/benchmark-state", headers=headers)).json()
        capacity = state["engine"]["loads"][0]["max_running_requests"]
        rows = []
        # Same physical engine and fixed work, alternate order to expose drift.
        for trial in range(3):
            for mode in (
                ["unmerged", "merged"] if trial % 2 == 0 else ["merged", "unmerged"]
            ):
                await control("unmerge_lora_weights")
                if mode == "merged":
                    await control("merge_lora_weights", {"lora_name": adapter})
                await control("flush_cache")
                before_counts = (
                    await client.get(gateway + "/benchmark-state", headers=headers)
                ).json()["gpu_lora"]
                began = time.monotonic()
                await asyncio.gather(*[request(prompt, 256) for _ in range(capacity)])
                seconds = time.monotonic() - began
                after_counts = (
                    await client.get(gateway + "/benchmark-state", headers=headers)
                ).json()["gpu_lora"]
                steps = sum(
                    after_counts.get(k, 0) - before_counts.get(k, 0)
                    for k in ["merged_decode_steps", "unmerged_decode_steps"]
                )
                running = after_counts.get(
                    "decode_batch_sequences", 0
                ) - before_counts.get("decode_batch_sequences", 0)
                rows.append(
                    dict(
                        trial=trial,
                        mode=mode,
                        batch=capacity,
                        output=256,
                        seconds=seconds,
                        observed_decode_steps=steps,
                        observed_mean_decode_batch=running / steps if steps else None,
                    )
                )
                print(rows[-1], flush=True)
        await control("unmerge_lora_weights")
        await control("flush_cache")
        vocabulary = sorted({t for p in source["prompts"] for t in p})
        prompts = [random.Random(144 + i).choices(vocabulary, k=8192) for i in range(4)]
        began = time.monotonic()
        await asyncio.gather(*[request(p, 1) for p in prompts])
        prefill_seconds = time.monotonic() - began
        profile = dict(
            unmerged_token_seconds=statistics.median(
                r["seconds"] / 256 for r in rows if r["mode"] == "unmerged"
            ),
            merged_token_seconds=statistics.median(
                r["seconds"] / 256 for r in rows if r["mode"] == "merged"
            ),
            prefill_tokens_per_second=4 * 8192 / prefill_seconds,
            merge_seconds=max(
                r["seconds"]
                for r in [r for r in timings if r["action"] == "merge_lora_weights"][1:]
            ),
            unmerge_seconds=max(
                r["seconds"] for r in timings if r["action"] == "unmerge_lora_weights"
            ),
            capacity=capacity,
            queue_buffer=4,
            max_wait_seconds=5,
            enabled=True,
        )
        await router.aclose()
        return dict(
            profile=profile,
            rows=rows,
            transitions=timings,
            replica=replica.replica_id,
            engine=state["engine"],
            exact_restore=True,
            merged_tokens=merged["output_ids"],
            unmerged_tokens=before["output_ids"],
            adapter=adapter,
            prefill_seconds=prefill_seconds,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads((args.root / "pools.json").read_text())[0]["pool"]
    sources = json.loads(args.clients.read_text())
    with modal.enable_output(), app.run(environment_name="lilo-deploy"):
        result = calibrate.remote(pool, sources)
    (args.root / "calibration.json").write_text(json.dumps(result, indent=2))
    (args.root / "merge-profile.json").write_text(
        json.dumps(result["profile"], indent=2)
    )


if __name__ == "__main__":
    main()
