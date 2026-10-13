"""Three-policy, fixed-work cache experiment on real inference GPUs.

Logical clients are independent closed loops, not synchronized waves. They all
replay the same seeded prompts and fixed output lengths under each policy.
"""

import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import httpx
import modal

from spindle.inference.sampling import TCP_KEEPALIVE_OPTIONS, sample_task
from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)
from spindle.providers.modal.lora_routing import create_router


app = modal.App("spindle-cache-routing-experiment")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .env({"MODAL_ENVIRONMENT": "lilo-deploy"})
    .add_local_python_source("spindle")
)


def make_trace(
    clients,
    *,
    streams=16,
    conversations=4,
    turns=4,
    prompt_tokens=8192,
    shared_tokens=7168,
    decode_tokens=512,
    seed=12026,
):
    vocabulary = sorted({t for c in clients for p in c["prompts"] for t in p})
    adapters = [(c["model_id"], v) for c in clients for v in c["versions"]]
    trace = []
    for adapter_index, (model, version) in enumerate(adapters):
        for stream in range(streams):
            conversation = stream % conversations
            prefix = random.Random(seed + adapter_index * 10000 + conversation).choices(
                vocabulary, k=shared_tokens
            )
            rng = random.Random(seed + adapter_index * 10000 + stream + 1_000_000)
            trace.append(
                {
                    "model_id": model,
                    "version": version,
                    "adapter_index": adapter_index,
                    "stream": stream,
                    "conversation": conversation,
                    "prompts": [
                        prefix
                        + rng.choices(vocabulary, k=prompt_tokens - shared_tokens)
                        for _ in range(turns)
                    ],
                    "decode_tokens": decode_tokens,
                }
            )
    random.Random(seed).shuffle(trace)
    return trace


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    max_containers=8,
    timeout=2400,
    nonpreemptible=True,
)
async def run_training_frontend(pool, arm, clients, config, startup, shard):
    """Two independent DAPO clients per real CPU container; one router per pool.

    Replay real prompts and saved policies. Recorded train/publication intervals
    are delays, not new optimizer work. One rollout is prefetched while that
    client's update proceeds, matching client.py without cross-client barriers.
    """
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = create_router(pool, gateway, target=64, slots=8)
    rows, rollouts = [], []
    await router.refresh()
    await startup.put.aio(
        {"shard": shard, "frontend": os.environ["MODAL_TASK_ID"]}, partition="ready"
    )
    start_at = await startup.get.aio(timeout=180, partition=f"start-{shard}")
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
    async with httpx.AsyncClient(
        headers=proxy_auth_headers(),
        timeout=240,
        trust_env=False,
        transport=httpx.AsyncHTTPTransport(
            limits=limits,
            socket_options=list(TCP_KEEPALIVE_OPTIONS),
            trust_env=False,
        ),
    ) as client:

        async def group(source, round_index, group_index, version):
            client_index = source["client_index"]
            problem = round_index * config["groups"] + group_index
            prompt = source["prompts"][problem % len(source["prompts"])]
            count = config["samples"]
            if client_index == 0 and config["hot_samples"]:
                count = config["hot_samples"]
            ident = f"client-{client_index}-round-{round_index}-group-{group_index}"
            task = {
                "request_id": ident,
                "sampling_session_id": f"dapo-{source['model_id']}-{version}",
                "model_id": source["model_id"],
                "publish_version": version,
                "payload": {
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt}]},
                    "num_samples": count,
                    "cache_affinity_key": f"problem-{problem % len(source['prompts'])}",
                    "sampling_params": {
                        "max_tokens": config["decode_tokens"],
                        "stop": [],
                        "temperature": 1.0,
                        "seed": 4242 + round_index * 100 + group_index,
                    },
                },
            }
            events, stats = [], {}
            began = time.time()
            result = await sample_task(
                task,
                gateway,
                client=client,
                lora_router=router if arm == "lora" else None,
                stats=stats,
                on_event=events.append,
                timeout=240,
                retry_timeout=900,
            )
            ended = time.time()
            attempts = [e for e in events if e.get("name") == "sample_attempt"]
            assert len(result["sequences"]) == count
            for index, sequence in enumerate(result["sequences"]):
                seq_attempts = [
                    e for e in attempts if e["attrs"]["sequence_index"] == index
                ]
                successful = [e for e in seq_attempts if e["attrs"].get("ok")]
                assert len(successful) == 1
                event = successful[0]
                meta = event["attrs"]["backend_timing"]
                assert (
                    meta["weight_version_start"]
                    == meta["weight_version_end"]
                    == version
                )
                assert (
                    len(sequence["tokens"])
                    == len(sequence["logprobs"])
                    == config["decode_tokens"]
                )
                assert sequence["stop_reason"] == "length"
                assert all(math.isfinite(x) for x in sequence["logprobs"])
                assert 0 <= meta["cached_tokens"] <= len(prompt)
                rows.append(
                    {
                        "id": f"{ident}-sample-{index}",
                        "client": client_index,
                        "turn": round_index,
                        "adapter_index": client_index * 2
                        + source["versions"].index(version),
                        "conversation": problem,
                        "model_id": source["model_id"],
                        "version": version,
                        "start": began - start_at,
                        "end": event["t1"] - start_at,
                        "seconds": event["t1"] - began,
                        "group_seconds": ended - began,
                        "prompt_tokens": len(prompt),
                        "decode_tokens": len(sequence["tokens"]),
                        "cached_tokens": meta["cached_tokens"],
                        "output_sha256": hashlib.sha256(
                            json.dumps(sequence["tokens"]).encode()
                        ).hexdigest(),
                        "stats": stats["sequence_stats"][index],
                        "attempts": seq_attempts,
                    }
                )

        async def rollout(source, round_index):
            version = source["versions"][0 if round_index < 2 else 1]
            began = time.time()
            groups = (
                1
                if source["client_index"] == 0 and config["hot_samples"]
                else config["groups"]
            )
            await asyncio.gather(
                *(group(source, round_index, g, version) for g in range(groups))
            )
            rollouts.append(
                {
                    "client": source["client_index"],
                    "round": round_index,
                    "version": version,
                    "start": began - start_at,
                    "end": time.time() - start_at,
                    "seconds": time.time() - began,
                    "groups": groups,
                }
            )
            print(
                f"{arm}: client {source['client_index']} rollout {round_index} finished in {time.time() - began:.1f}s",
                flush=True,
            )

        async def training_client(source):
            await asyncio.sleep(
                max(
                    0,
                    start_at
                    + source["client_index"] * config["stagger_seconds"]
                    - time.time(),
                )
            )
            await rollout(source, 0)
            if config["rounds"] > 1:
                # The next rollout starts immediately while this client's
                # recorded update/publication interval elapses independently.
                await asyncio.gather(
                    rollout(source, 1), asyncio.sleep(source["update_seconds"])
                )
            if config["rounds"] > 2:
                await rollout(source, 2)

        try:
            await asyncio.gather(*(training_client(source) for source in clients))
        finally:
            await router.aclose()
    return {
        "requests": rows,
        "rollouts": rollouts,
        "frontend": os.environ["MODAL_TASK_ID"],
    }


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    region="us-west",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=3600,
    nonpreemptible=True,
)
async def run_arm(pool, arm, clients, config):
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    routers = [create_router(pool, gateway, target=64, slots=8) for _ in range(8)]
    deadline = time.monotonic() + 1100
    async with httpx.AsyncClient(
        headers=proxy_auth_headers(),
        timeout=900,
        trust_env=False,
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
    ) as client:
        while time.monotonic() < deadline:
            await routers[0]._refresh()
            if len(routers[0].replicas) == 8:
                break
            print(f"{arm}: {len(routers[0].replicas)}/8 replicas ready", flush=True)
            await asyncio.sleep(5)
        if len(routers[0].replicas) != 8:
            raise RuntimeError(f"{arm}: eight healthy replicas did not start")
        replicas = dict(routers[0].replicas)

        async def inspect():
            async def one(replica):
                response = await client.get(
                    gateway + "/benchmark-state",
                    headers={"Modal-Flash-Upstream": replica.upstream},
                    timeout=10,
                )
                response.raise_for_status()
                state = response.json()
                assert state["replica"] == replica.replica_id
                return state

            return await asyncio.gather(*(one(r) for r in replicas.values()))

        # All arms begin with every adapter CPU-registered on every replica.
        # GPU slots hold the last seven adapters plus the base model; the trace
        # therefore isolates placement and GPU/KV churn from registration I/O.
        async def warm(replica):
            for source in clients:
                for version in source["versions"]:
                    response = await client.post(
                        gateway + "/generate",
                        headers={"Modal-Flash-Upstream": replica.upstream},
                        json={
                            "input_ids": source["prompts"][0][:16],
                            "weight_run_id": source["model_id"],
                            "weight_version": {
                                "exact_version": version,
                                "min_version": None,
                            },
                            "sampling_params": {
                                "max_new_tokens": 1,
                                "ignore_eos": True,
                                "temperature": 0,
                            },
                            "return_logprob": True,
                        },
                    )
                    response.raise_for_status()
                    meta = response.json()["meta_info"]
                    assert (
                        meta["weight_version_start"]
                        == meta["weight_version_end"]
                        == version
                    )
            # Engine health probes also use the base-model slot. Materialize it
            # explicitly so every trial starts with exactly the same GPU set.
            response = await client.post(
                gateway + "/generate",
                headers={"Modal-Flash-Upstream": replica.upstream},
                json={
                    "input_ids": [0],
                    "sampling_params": {
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
            )
            response.raise_for_status()
            response = await client.post(
                gateway + "/benchmark-flush-cache",
                headers={"Modal-Flash-Upstream": replica.upstream},
            )
            response.raise_for_status()

        await asyncio.gather(*(warm(r) for r in replicas.values()))
        print(f"{arm}: all 32 adapters registered on all eight replicas", flush=True)
        for router in routers:
            await router.refresh()
        baseline = await inspect()
        assert all(s["active"] == 0 and s["gpu_lora"] for s in baseline)
        assert all(s["adapter_loads"] == len(clients) * 2 for s in baseline)
        expected_residents = sorted(
            f"{c['model_id']}/weight_v{v:06d}"
            for c, v in [(c, v) for c in clients for v in c["versions"]][-7:]
        )
        assert all(
            s["gpu_lora"]["resident_adapters"] == expected_residents for s in baseline
        )
        training = config.get("workload") == "dapo"
        trace = (
            {"clients": clients, "config": config}
            if training
            else make_trace(clients, **config)
        )
        trace_hash = hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        requests, observations = [], []
        started = time.monotonic()
        wall_started = time.time()
        finished = asyncio.Event()

        async def observe():
            await asyncio.sleep(max(0, started - time.monotonic()))
            while not finished.is_set():
                try:
                    states = await inspect()
                    observations.append(
                        {"elapsed": time.monotonic() - started, "replicas": states}
                    )
                except Exception as exc:
                    observations.append(
                        {"elapsed": time.monotonic() - started, "error": str(exc)}
                    )
                try:
                    await asyncio.wait_for(finished.wait(), timeout=2)
                except TimeoutError:
                    pass

        async def logical_client(index, stream):
            # A bounded launch ramp avoids a single event-loop dispatch artifact.
            await asyncio.sleep(index / len(trace) * 2)
            for turn, prompt in enumerate(stream["prompts"]):
                ident = f"trace-{index}-{turn}"
                task = {
                    "request_id": ident,
                    "sampling_session_id": f"adapter-{stream['adapter_index']}",
                    "engine_definition_id": "cache-experiment",
                    "model_id": stream["model_id"],
                    "publish_version": stream["version"],
                    "payload": {
                        "prompt": {
                            "chunks": [{"type": "encoded_text", "tokens": prompt}]
                        },
                        "num_samples": 1,
                        "sampling_params": {
                            "max_tokens": stream["decode_tokens"],
                            "stop": [],
                            "temperature": 0,
                            "seed": 12026 + index * 100 + turn,
                        },
                    },
                }
                if arm != "default":
                    task["payload"]["cache_affinity_key"] = (
                        f"conversation-{stream['conversation']}"
                    )
                events, stats = [], {}
                request_started = time.monotonic()
                result = await sample_task(
                    task,
                    gateway,
                    client=client,
                    lora_router=routers[index % 8] if arm == "lora" else None,
                    stats=stats,
                    on_event=events.append,
                    timeout=240,
                    retry_timeout=900,
                )
                sequence = result["sequences"][0]
                assert (
                    len(sequence["tokens"])
                    == len(sequence["logprobs"])
                    == stream["decode_tokens"]
                )
                assert sequence["stop_reason"] == "length"
                assert all(math.isfinite(x) for x in sequence["logprobs"])
                assert (
                    stats["version_served_start"]
                    == stats["version_served_end"]
                    == stream["version"]
                )
                assert 0 <= result["prompt_cache_hit_tokens"] <= len(prompt)
                requests.append(
                    {
                        "id": ident,
                        "client": index,
                        "turn": turn,
                        "adapter_index": stream["adapter_index"],
                        "conversation": stream["conversation"],
                        "model_id": stream["model_id"],
                        "version": stream["version"],
                        "start": request_started - started,
                        "end": time.monotonic() - started,
                        "seconds": time.monotonic() - request_started,
                        "prompt_tokens": len(prompt),
                        "decode_tokens": len(sequence["tokens"]),
                        "cached_tokens": result["prompt_cache_hit_tokens"],
                        "output_sha256": hashlib.sha256(
                            json.dumps(sequence["tokens"]).encode()
                        ).hexdigest(),
                        "stats": stats,
                        "attempts": [
                            e for e in events if e.get("name") == "sample_attempt"
                        ],
                    }
                )
                if len(requests) % 128 == 0:
                    print(
                        f"{arm}: completed {len(requests)}/{len(trace) * config['turns']} requests in {time.monotonic() - started:.1f}s",
                        flush=True,
                    )

        async def frontend(shard, startup):
            result = await run_training_frontend.remote.aio(
                pool, arm, clients[shard::8], config, startup, shard
            )
            requests.extend(result["requests"])
            return result

        error = None
        results = []
        # Coordination occurs once at startup, never on sample requests or
        # between training rounds. The queue is destroyed when this arm ends.
        async with (
            modal.Queue.ephemeral() if training else nullcontext(None)
        ) as startup:
            tasks = (
                [asyncio.create_task(frontend(i, startup)) for i in range(8)]
                if training
                else [
                    asyncio.create_task(logical_client(i, s))
                    for i, s in enumerate(trace)
                ]
            )
            observer = None
            try:
                if training:
                    async with asyncio.timeout(180):
                        ready = [
                            await startup.get.aio(timeout=180, partition="ready")
                            for _ in range(8)
                        ]
                    assert {r["shard"] for r in ready} == set(range(8))
                    assert len({r["frontend"] for r in ready}) == 8
                    wall_started = time.time() + 2
                    started = time.monotonic() + 2
                    await asyncio.gather(
                        *(
                            startup.put.aio(wall_started, partition=f"start-{i}")
                            for i in range(8)
                        )
                    )
                    print(f"{arm}: all eight CPU frontends ready", flush=True)
                observer = asyncio.create_task(observe())
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2400)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"{arm}: FAILED: {error}", flush=True)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                finished.set()
                if observer is not None:
                    await observer
        elapsed = time.monotonic() - started
        final = await inspect()
        await asyncio.gather(*(router.aclose() for router in routers))
        if error is None:
            assert all(s["active"] == 0 for s in final)
        return {
            "arm": arm,
            "pool": pool,
            "config": config,
            "trace_sha256": trace_hash,
            "wall_started": wall_started,
            "seconds": elapsed,
            "requests": requests,
            "baseline": baseline,
            "final": final,
            "observations": observations,
            "error": error,
            "frontends": [r["frontend"] for r in results] if training else [],
            "rollouts": [s for r in results for s in r["rollouts"]] if training else [],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--clients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--conversations", type=int, default=4)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--shared-tokens", type=int, default=7168)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12026)
    parser.add_argument(
        "--workload", choices=("synthetic", "dapo"), default="synthetic"
    )
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--stagger-seconds", type=float, default=5)
    parser.add_argument("--hot-samples", type=int, default=0)
    parser.add_argument("--keep-pools", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("default", "lora", "kv"))
    args = parser.parse_args()
    pools = json.loads(args.pools.read_text())
    if args.arms:
        pools = [next(p for p in pools if p["arm"] == arm) for arm in args.arms]
    clients = json.loads(args.clients.read_text())
    config = {
        k: getattr(args, k)
        for k in (
            "streams",
            "conversations",
            "turns",
            "prompt_tokens",
            "shared_tokens",
            "decode_tokens",
            "seed",
        )
    }
    if args.workload == "dapo":
        config = dict(
            workload="dapo",
            groups=args.groups,
            samples=args.samples,
            rounds=args.rounds,
            stagger_seconds=args.stagger_seconds,
            hot_samples=args.hot_samples,
            decode_tokens=args.decode_tokens,
        )
        per_round = len(clients) * args.groups * args.samples
        if args.hot_samples:
            per_round += args.hot_samples - args.groups * args.samples
        config["expected_requests"] = per_round * args.rounds
        config["max_concurrency"] = per_round
    args.output.mkdir(parents=True, exist_ok=True)

    async def run():
        async def one(pool):
            result = await run_arm.remote.aio(
                pool["pool"], pool["arm"], clients, config
            )
            (args.output / f"{pool['arm']}.json").write_text(json.dumps(result) + "\n")
            return result

        if args.sequential or len({p["pool"] for p in pools}) < len(pools):
            results = []
            for pool in pools:
                result = await one(pool)
                results.append(result)
                if result.get("error"):
                    break
            return results
        return await asyncio.gather(*(one(p) for p in pools))

    try:
        with modal.enable_output(), app.run():
            results = asyncio.run(run())
            if any(result.get("error") for result in results):
                raise RuntimeError("Incomplete benchmark; inspect saved arm results")
    finally:
        if not args.keep_pools:
            for pool in sorted({p["pool"] for p in pools}):
                subprocess.run(
                    [
                        "modal",
                        "app",
                        "stop",
                        "-y",
                        pool,
                        "--env",
                        "lilo-deploy",
                    ],
                    check=False,
                )


if __name__ == "__main__":
    main()
