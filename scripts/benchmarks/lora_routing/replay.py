"""Controlled inference replay of adapters from a completed training run.

This supplements the independent training benchmark; it does not measure training
throughput. Both arms receive identical prompts, weights, seeds and arrivals.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import modal

from spindle.inference.sampling import sample_task
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)
from spindle.providers.modal.lora_routing import create_router
from spindle.providers.modal.fft_pool import proxy_auth_headers

app = modal.App("spindle-lora-routing-replay")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .env({"MODAL_ENVIRONMENT": "lilo-deploy"})
    .add_local_python_source("spindle")
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=1800,
)
async def replay(pool, mode, clients, waves, max_tokens):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    routers = [create_router(pool, gateway, target=64, slots=8) for _ in range(8)]
    deadline = time.monotonic() + 600
    last_count = None
    while time.monotonic() < deadline:
        await routers[0]._refresh()
        if len(routers[0].replicas) != last_count:
            last_count = len(routers[0].replicas)
            print(f"{mode}: {last_count}/8 healthy replicas", flush=True)
        if len(routers[0].replicas) == 8:
            break
        await asyncio.sleep(2)
    if len(routers[0].replicas) != 8:
        raise RuntimeError("The replay needs all eight healthy replicas")
    for router in routers[1:]:
        await router.refresh()
    output = []

    async def sample(client_index, group_index, wave, warmup=False):
        client = clients[client_index]
        version = client["versions"][wave % 2]
        task = {
            "request_id": f"replay-{wave}-{client_index}-{group_index}",
            "sampling_session_id": f"replay-{client_index}-{version}",
            "engine_definition_id": "replay",
            "model_id": client["model_id"],
            "publish_version": version,
            "payload": {
                "prompt": {
                    "chunks": [
                        {
                            "type": "encoded_text",
                            "tokens": client["prompts"][group_index],
                        }
                    ]
                },
                "num_samples": 1 if warmup else 8,
                "sampling_params": {
                    "max_tokens": 16 if warmup else max_tokens,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "seed": 4242 + group_index,
                    "stop": client["stops"],
                },
            },
        }
        stats = {}
        start = time.monotonic()
        result = await sample_task(
            task,
            gateway,
            headers=proxy_auth_headers(),
            lora_router=routers[(client_index * 8 + group_index) % 8]
            if mode == "affinity"
            else None,
            stats=stats,
            retry_timeout=90,
            timeout=900,
        )
        for sequence in result["sequences"]:
            assert len(sequence["tokens"]) == len(sequence["logprobs"]) > 0
            assert all(math.isfinite(x) for x in sequence["logprobs"])
        return {
            "client": client_index,
            "group": group_index,
            "version": version,
            "seconds": time.monotonic() - start,
            "tokens": sum(len(s["tokens"]) for s in result["sequences"]),
            "reroutes": stats["reroutes"],
        }

    await asyncio.gather(*(sample(i, 0, 0, warmup=True) for i in range(len(clients))))
    print(f"{mode}: adapter warmup complete; starting measured waves", flush=True)
    for wave in range(waves):
        started = time.monotonic()
        groups = await asyncio.gather(
            *(sample(i, g, wave) for g in range(8) for i in range(len(clients)))
        )
        seconds = time.monotonic() - started
        tokens = sum(g["tokens"] for g in groups)
        row = {
            "wave": wave,
            "versions": sorted({g["version"] for g in groups}),
            "seconds": seconds,
            "tokens": tokens,
            "tokens_per_second": tokens / seconds,
            "group_mean_seconds": statistics.mean(g["seconds"] for g in groups),
            "group_p95_seconds": sorted(g["seconds"] for g in groups)[
                math.ceil(len(groups) * 0.95) - 1
            ],
            "groups": groups,
        }
        output.append(row)
        print(
            json.dumps(
                {"mode": mode, **{k: v for k, v in row.items() if k != "groups"}}
            ),
            flush=True,
        )
    return {
        "mode": mode,
        "pool": pool,
        "waves": output,
        "workload_sha256": hashlib.sha256(
            json.dumps(clients, sort_keys=True).encode()
        ).hexdigest(),
        "max_tokens": max_tokens,
    }


def load_clients(source, bulletin_volume):
    clients = []
    bulletin = modal.Volume.from_name(bulletin_volume, version=2)
    for folder in sorted(source.glob("client-*")):
        if not folder.is_dir():
            continue
        events = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        assert any(r["event"] == "completed" for r in events)
        manifest = json.loads((folder / "manifest.json").read_text())
        model_id = next(r["model_id"] for r in events if r["event"] == "client_created")
        # Publication versions count operations, not optimizer steps. Inspect
        # immutable manifests rather than guessing a version from the step index.
        versions = sorted(
            int(Path(entry.path).name.removeprefix("weight_v"))
            for entry in bulletin.listdir(model_id)
            if Path(entry.path).name.startswith("weight_v")
        )[1:3]
        assert len(versions) == 2, f"Missing trained snapshots for {model_id}"
        snapshots = []
        for version in versions:
            identity = f"{model_id}/weight_v{version:06d}"
            snapshot = json.loads(
                b"".join(bulletin.read_file(identity + "/snapshot.json"))
            )
            assert snapshot["ref"] == identity
            snapshots.append(snapshot)
        clients.append(
            {
                "model_id": model_id,
                "versions": versions,
                "snapshots": snapshots,
                "prompts": [
                    json.load(gzip.open(folder / f"group-{g + 32:06d}.json.gz", "rt"))[
                        "prompt"
                    ]
                    for g in range(8)
                ],
                "stops": manifest["stop_token_ids"],
            }
        )
    assert len(clients) == 16

    return clients


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pools", nargs=2, required=True, help="naive then affinity")
    p.add_argument("--waves", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--bulletin-volume", default="routing27-bulletin-1011")
    args = p.parse_args()

    async def run():
        return await asyncio.gather(
            *(
                replay.remote.aio(pool, mode, clients, args.waves, args.max_tokens)
                for pool, mode in zip(args.pools, ("naive", "affinity"), strict=True)
            )
        )

    try:
        clients = load_clients(args.source, args.bulletin_volume)
        with modal.enable_output(), app.run():
            results = asyncio.run(run())
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        cleanup_errors = []
        for pool in args.pools:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "modal",
                        "app",
                        "stop",
                        "-y",
                        pool,
                        "--env",
                        "lilo-deploy",
                    ],
                    check=True,
                    timeout=120,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                cleanup_errors.append(f"{pool}: {exc}")
        if cleanup_errors:
            raise RuntimeError("Pool cleanup failed: " + "; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
