"""Three-policy, fixed-work cache experiment on real inference GPUs.

Logical clients are independent closed loops, not synchronized waves. They all
replay the same seeded prompts and fixed output lengths under each policy.
"""

import argparse
import asyncio
from contextlib import AsyncExitStack, nullcontext
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
import h2_diagnostics

from spindle.inference.sampling import keepalive_transport, sample_task
from cohort_prefill import prepare_cohort
from spindle.providers.modal.fft_pool import proxy_auth_headers
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)
from spindle.providers.modal.lora_routing import create_router
from spindle.inference.routing_simple import SIMPLE_POLICIES


app = modal.App("spindle-cache-routing-experiment")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "h2>=4,<5")
    .env({"MODAL_ENVIRONMENT": "lilo-deploy"})
    .add_local_python_source("spindle", "cohort_prefill", "h2_diagnostics")
)


async def gather_work(*coroutines):
    """Drain sibling cancellation before closing shared clients after failure."""
    tasks = [asyncio.create_task(c) for c in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def make_router(pool, gateway, arm, config):
    policy = arm.removesuffix("-cohort")
    measured = policy.startswith("work") or policy in {
        "domains",
        "balanced",
        "stable",
        "capacity",
        "capacity-random",
        *SIMPLE_POLICIES,
    }
    return create_router(
        pool,
        gateway,
        target=64,
        slots=8,
        work_rates=config.get("work_rates") if measured else None,
        adapter_costs=policy != "work-no-adapter",
        domains=policy in {"domains", "balanced", "stable"},
        balanced_domains=policy in {"balanced", "stable"},
        stable_domains=policy == "stable",
        prefill_once=arm.endswith("-cohort"),
        simple_policy=policy if policy in SIMPLE_POLICIES else None,
        capacity_aware=policy.startswith("capacity"),
        capacity_tie_seed=os.environ["MODAL_TASK_ID"]
        if policy == "capacity-random"
        else None,
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


def long_prompt(source, round_index, group_index):
    client_index = source["client_index"]
    vocabulary = source["vocabulary"]
    shared = random.Random(
        source.get("prompt_seed", 12026) + client_index * 1000 + group_index
    ).choices(vocabulary, k=7168)
    suffix = random.Random(
        source.get("prompt_seed", 12026)
        + 12000
        + client_index * 10000
        + round_index * 100
        + group_index
    ).choices(vocabulary, k=1024)
    return shared + suffix


async def claim_execution(executions, key):
    """A benchmark input cannot safely replay after a worker disappears."""
    claimed = await executions.put.aio(
        key, os.environ["MODAL_TASK_ID"], skip_if_exists=True
    )
    if not claimed:
        raise RuntimeError(
            f"Benchmark worker was rescheduled for {key}; preserve the interrupted arm"
        )


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    region="us-east",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    max_containers=8,
    timeout=2400,
    nonpreemptible=True,
)
async def run_training_frontend(pool, arm, clients, config, startup, shard, executions):
    """Two independent DAPO clients per real CPU container; one router per pool.

    Replay real prompts and saved policies. Recorded train/publication intervals
    are delays, not new optimizer work. One rollout is prefetched while that
    client's update proceeds, matching client.py without cross-client barriers.
    """
    await claim_execution(executions, f"{pool}/{arm}/frontend-{shard}")
    policy = arm.removesuffix("-cohort")
    h2_diagnostics.enable()
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    router = make_router(pool, gateway, arm, config)
    rows, rollouts, prefills = [], [], []
    await router.refresh()
    await startup.put.aio(
        {"shard": shard, "frontend": os.environ["MODAL_TASK_ID"]}, partition="ready"
    )
    start_at = await startup.get.aio(timeout=180, partition=f"start-{shard}")

    async def group(
        source,
        round_index,
        group_index,
        version,
        *,
        history=None,
        trajectory=None,
        dialogue_turn=0,
    ):
        client_index = source["client_index"]
        problem = (
            0 if config.get("repeat_prefixes") else round_index * config["groups"]
        ) + group_index
        prompt = source["prompts"][problem % len(source["prompts"])]
        if config.get("long_prefix"):
            prompt = long_prompt(source, round_index, group_index)
        if history is not None:
            prompt = history
        count = config["samples"]
        if client_index == 0 and config["hot_samples"]:
            count = config["hot_samples"]
        elif client_index < config.get("large_group_clients", 0):
            count = config["groups"] * config["samples"]
        if trajectory is not None:
            count = 1
        ident = f"client-{client_index}-round-{round_index}-group-{group_index}"
        if dialogue_turn:
            ident += f"-trajectory-{trajectory}-turn-{dialogue_turn}"
        task = {
            "request_id": ident,
            "sampling_session_id": f"dapo-{source['model_id']}-{version}",
            "model_id": source["model_id"],
            "publish_version": version,
            "payload": {
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt}]},
                "num_samples": count,
                "cache_affinity_key": config.get("affinity_salt", "")
                + f"problem-{problem % len(source['prompts'])}",
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
        if arm.endswith("-cohort"):
            # Match the short transport lifetime used by normal sample tasks.
            # Long-lived multiplexed preparation connections failed under burst
            # pressure; HTTP/1 preparation also stalled after engines drained.
            async with httpx.AsyncClient(
                headers=proxy_auth_headers(),
                transport=keepalive_transport(http2=config.get("http2", False)),
                timeout=240,
                trust_env=False,
            ) as preparation_client:
                await prepare_cohort(
                    task,
                    gateway,
                    preparation_client,
                    router,
                    prefills,
                    native=policy == "kv",
                    timeout=config.get("request_timeout", 240),
                )
        result = await sample_task(
            task,
            gateway,
            headers=proxy_auth_headers(),
            transport=keepalive_transport(http2=config.get("http2", False)),
            lora_router=router if policy not in {"kv", "default"} else None,
            stats=stats,
            on_event=events.append,
            timeout=config.get("request_timeout", 240),
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
            if "admission_limit" in event["attrs"].get("routing", {}):
                assert event["attrs"]["admission_checked"], (
                    "Admission header was not enforced"
                )
            meta = event["attrs"]["backend_timing"]
            assert meta["weight_version_start"] == meta["weight_version_end"] == version
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
                    "dialogue_turn": dialogue_turn,
                    "trajectory": trajectory if trajectory is not None else index,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(prompt).encode()
                    ).hexdigest(),
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

        if dialogue_turn + 1 < config.get("dialogue_turns", 1):

            async def continuation(index, sequence):
                branch = index if trajectory is None else trajectory
                # A fixed-length synthetic tool reply follows the actual model
                # response. Each branch depends on its own preceding response.
                tool_reply = random.Random(
                    8000
                    + client_index * 10000
                    + round_index * 1000
                    + group_index * 100
                    + branch * 10
                    + dialogue_turn
                ).choices(source["vocabulary"], k=256)
                await group(
                    source,
                    round_index,
                    group_index,
                    version,
                    history=prompt + sequence["tokens"] + tool_reply,
                    trajectory=branch,
                    dialogue_turn=dialogue_turn + 1,
                )

            await gather_work(
                *(continuation(i, seq) for i, seq in enumerate(result["sequences"]))
            )

    async def rollout(source, round_index):
        version = source["versions"][
            0 if round_index < 2 or not config.get("version_turnover", True) else 1
        ]
        began = time.time()
        groups = (
            1
            if source["client_index"] < config.get("large_group_clients", 0)
            or (source["client_index"] == 0 and config["hot_samples"])
            else config["groups"]
        )
        await gather_work(
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
        for round_index in range(1, config["rounds"]):
            if (
                config.get("duration_seconds")
                and time.time() - start_at >= config["duration_seconds"]
            ):
                break
            # Preserve independent prefetch/update overlap. Saved checkpoint
            # arrivals replay publication; these sleeps are not optimizer work.
            await gather_work(
                rollout(source, round_index),
                asyncio.sleep(source["update_seconds"]),
            )

    try:
        await gather_work(*(training_client(source) for source in clients))
    finally:
        await router.aclose()
    return {
        "requests": rows,
        "rollouts": rollouts,
        "prefills": prefills,
        "frontend": os.environ["MODAL_TASK_ID"],
        "prefix_homes": dict(router.prefix_homes),
        "lookup_metrics": dict(
            count=router.lookup_count,
            failures=router.lookup_failures,
            seconds=router.lookup_seconds,
            max_seconds=router.lookup_max_seconds,
        ),
    }


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    region="us-east",
    secrets=[modal.Secret.from_name("lilo-proxy")],
    timeout=3600,
    nonpreemptible=True,
)
async def run_arm(pool, arm, clients, config, executions):
    await claim_execution(executions, f"{pool}/{arm}/controller")
    policy = arm.removesuffix("-cohort")
    gateway = await modal.Server.from_name(pool, "Server").get_url.aio()
    routers = [make_router(pool, gateway, arm, config) for _ in range(8)]
    deadline = time.monotonic() + 1100
    async with (
        httpx.AsyncClient(
            headers=proxy_auth_headers(),
            http2=config.get("http2", False),
            timeout=900,
            trust_env=False,
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
        ) as client,
        AsyncExitStack() as client_stack,
    ):
        # Short setup responses can be reset by the Flash HTTP/2 path. Keep
        # catalog reset and prefix warming on HTTP/1.1, outside the timed arm;
        # generation and observations retain the selected measured transport.
        setup_client = await client_stack.enter_async_context(
            httpx.AsyncClient(
                headers=proxy_auth_headers(),
                http2=False,
                timeout=900,
                trust_env=False,
            )
        )
        # HTTPcore caps one HTTP/2 connection at 100 streams even when the
        # server advertises more. Keep generation shards and observations on
        # separate connections so a 512-stream replay reaches the engines.
        generation_clients = (
            [
                await client_stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=proxy_auth_headers(),
                        http2=True,
                        timeout=900,
                        trust_env=False,
                        limits=httpx.Limits(
                            max_connections=128, max_keepalive_connections=128
                        ),
                    )
                )
                for _ in range(config["http2_client_shards"])
            ]
            if config.get("http2_client_shards")
            else [client]
        )
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

        # Remove the previous trial's catalog, including adapters absent from
        # this workload. Otherwise domain placement depends on experiment order.
        previous = {s["replica"]: s for s in await inspect()}

        # All arms begin with every initial adapter CPU-registered on every replica.
        # GPU slots hold the last seven adapters plus the base model; the trace
        # therefore isolates placement and GPU/KV churn from registration I/O.
        async def warm(replica):
            registered = previous[replica.replica_id]["work"]["cpu_adapters"]
            if registered:
                response = await setup_client.post(
                    gateway + "/benchmark-unregister",
                    headers={"Modal-Flash-Upstream": replica.upstream},
                    json={"adapters": registered},
                )
                response.raise_for_status()
            for source in clients:
                for version in (
                    source["versions"][:1]
                    if config.get("cold_versions")
                    else source["versions"]
                ):
                    response = await setup_client.post(
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
            response = await setup_client.post(
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
            response = await setup_client.post(
                gateway + "/benchmark-flush-cache",
                headers={"Modal-Flash-Upstream": replica.upstream},
            )
            response.raise_for_status()

        await asyncio.gather(*(warm(r) for r in replicas.values()))
        print(f"{arm}: adapter reset completed on all eight replicas", flush=True)
        warmup = []
        if config.get("warm_prefixes"):

            async def warm_client(source):
                groups = (
                    1
                    if source["client_index"] < config.get("large_group_clients", 0)
                    else config["groups"]
                )
                for group in range(groups):
                    for turn in (-3, -2, -1):
                        prompt = long_prompt(source, turn, group)
                        task = dict(
                            request_id=f"warm-{source['client_index']}-{group}-{turn}",
                            sampling_session_id=f"dapo-{source['model_id']}-{source['versions'][0]}",
                            model_id=source["model_id"],
                            publish_version=source["versions"][0],
                            payload=dict(
                                prompt={
                                    "chunks": [
                                        {"type": "encoded_text", "tokens": prompt}
                                    ]
                                },
                                num_samples=1,
                                cache_affinity_key=config.get("affinity_salt", "")
                                + f"problem-{group % len(source['prompts'])}",
                                sampling_params=dict(
                                    max_tokens=1, stop=[], temperature=0
                                ),
                            ),
                        )
                        result = await sample_task(task, gateway, client=setup_client)
                        warmup.append(
                            dict(
                                client=source["client_index"],
                                group=group,
                                turn=turn,
                                cached_tokens=result["prompt_cache_hit_tokens"],
                            )
                        )

            await asyncio.gather(*(warm_client(c) for c in clients))
            print(f"{arm}: serialized prefix warmup complete", flush=True)
        for router in routers:
            await router.refresh()
        baseline = await inspect()
        assert all(s["active"] == 0 and s["gpu_lora"] for s in baseline)
        if policy.startswith("capacity"):
            assert all("adapter_decode_maxima" in s["work"] for s in baseline)
            assert all(
                load["max_running_requests"]
                == config["work_rates"]["decode_concurrency"]
                for s in baseline
                for load in s["engine"]["loads"]
            ), "Capacity calibration does not match the deployed engine"
        registered_versions = [
            (c, v)
            for c in clients
            for v in (
                c["versions"][:1] if config.get("cold_versions") else c["versions"]
            )
        ]
        expected_residents = sorted(
            f"{c['model_id']}/weight_v{v:06d}" for c, v in registered_versions[-7:]
        )
        assert config.get("warm_prefixes") or all(
            s["gpu_lora"]["resident_adapters"] == expected_residents for s in baseline
        )
        training = config.get("workload") == "dapo"
        trace = (
            {"clients": clients, "config": config}
            if training
            else make_trace(
                clients,
                **{
                    k: config[k]
                    for k in (
                        "streams",
                        "conversations",
                        "turns",
                        "prompt_tokens",
                        "shared_tokens",
                        "decode_tokens",
                        "seed",
                    )
                },
            )
        )
        if not training and config.get("arrival_rate", 0):
            expanded = [
                stream | {"prompts": [prompt], "turn_offset": turn}
                for stream in trace
                for turn, prompt in enumerate(stream["prompts"])
            ]
            rng = random.Random(config["seed"] + 17)
            rng.shuffle(expanded)
            arrival = 0.0
            for stream in expanded:
                arrival += rng.expovariate(config["arrival_rate"])
                stream["arrival"] = arrival
            trace = expanded
        trace_hash = hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        workload_spec = {
            "clients": clients,
            "config": {
                k: v
                for k, v in config.items()
                if k not in {"source_sha256", "work_rates"}
            },
        }
        workload_hash = hashlib.sha256(
            json.dumps(workload_spec, sort_keys=True, separators=(",", ":")).encode()
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
            await asyncio.sleep(
                max(
                    0,
                    started
                    + stream.get("arrival", index / len(trace) * 2)
                    - time.monotonic(),
                )
            )
            for turn, prompt in enumerate(stream["prompts"]):
                turn += stream.get("turn_offset", 0)
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
                        config.get("affinity_salt", "")
                        + f"conversation-{stream['conversation']}"
                    )
                events, stats = [], {}
                request_started = time.monotonic()
                result = await sample_task(
                    task,
                    gateway,
                    client=generation_clients[index % len(generation_clients)],
                    lora_router=routers[index % 8]
                    if policy not in {"kv", "default"}
                    else None,
                    stats=stats,
                    on_event=events.append,
                    timeout=config.get("request_timeout", 240),
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
                for event in events:
                    attrs = event.get("attrs", {})
                    if attrs.get("ok") and "admission_limit" in attrs.get(
                        "routing", {}
                    ):
                        assert attrs["admission_checked"], (
                            "Admission header was not enforced"
                        )
                requests.append(
                    {
                        "id": ident,
                        "client": index,
                        "turn": turn,
                        "adapter_index": stream["adapter_index"],
                        "conversation": stream["conversation"],
                        "model_id": stream["model_id"],
                        "version": stream["version"],
                        "scheduled_arrival": stream.get("arrival"),
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
                        f"{arm}: completed {len(requests)}/{sum(len(s['prompts']) for s in trace)} requests in {time.monotonic() - started:.1f}s",
                        flush=True,
                    )

        async def frontend(shard, startup):
            result = await run_training_frontend.remote.aio(
                pool, arm, clients[shard::8], config, startup, shard, executions
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
        if error is None:
            # Native load counters may lag the last completed decode batch.
            # Settle the final observation after freezing the measured timer.
            drain_deadline = time.monotonic() + 10
            while any(
                s["active"]
                or any(
                    v["num_running_reqs"] or v["num_waiting_reqs"]
                    for v in s["engine"]["loads"]
                )
                for s in final
            ):
                if time.monotonic() >= drain_deadline:
                    raise RuntimeError(
                        "Completed replay did not drain native engine queues"
                    )
                await asyncio.sleep(0.1)
                final = await inspect()
        await asyncio.gather(*(router.aclose() for router in routers))
        if error is None:
            assert all(s["active"] == 0 for s in final)
            if not training:
                capacity = sum(
                    load["max_running_requests"]
                    for state in baseline
                    for load in state["engine"]["loads"]
                )
                assert (
                    max(
                        sum(s["active"] for s in o["replicas"])
                        for o in observations
                        if "replicas" in o
                    )
                    >= capacity
                ), "General-serving replay did not reach pool admission pressure"
            protocol = "http2." if config.get("http2") else "http11."
            assert all(
                any(t["event"].startswith(protocol) for t in a["attrs"]["http_trace"])
                for row in requests
                for a in row["attempts"]
                if a["attrs"].get("ok")
            ), "Every successful attempt must use the configured HTTP version"
        return {
            "arm": arm,
            "pool": pool,
            "config": config,
            "trace_sha256": trace_hash,
            "workload_sha256": workload_hash,
            "wall_started": wall_started,
            "seconds": elapsed,
            "requests": requests,
            "baseline": baseline,
            "final": final,
            "observations": observations,
            "warmup": warmup,
            "prefills": [p for r in results for p in r["prefills"]] if training else [],
            "error": error,
            "frontends": [r["frontend"] for r in results] if training else [],
            "rollouts": [s for r in results for s in r["rollouts"]] if training else [],
            "native_prefix_homes": [r["prefix_homes"] for r in results]
            if training
            else [],
            "lookup_metrics": [r["lookup_metrics"] for r in results]
            if training
            else [
                dict(
                    count=r.lookup_count,
                    failures=r.lookup_failures,
                    seconds=r.lookup_seconds,
                    max_seconds=r.lookup_max_seconds,
                )
                for r in routers
            ],
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
        "--vary-placement",
        action="store_true",
        help="Seed client order and affinity identities as well as prompt contents",
    )
    parser.add_argument("--arrival-rate", type=float, default=0)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--http2", action="store_true")
    parser.add_argument(
        "--workload", choices=("synthetic", "dapo"), default="synthetic"
    )
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--dialogue-turns", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--duration-seconds", type=float, default=0)
    parser.add_argument("--long-prefix", action="store_true")
    parser.add_argument("--warm-prefixes", action="store_true")
    parser.add_argument("--repeat-prefixes", action="store_true")
    parser.add_argument("--fixed-version", action="store_true")
    parser.add_argument("--cold-versions", action="store_true")
    parser.add_argument("--work-rates", type=Path)
    parser.add_argument("--stagger-seconds", type=float, default=5)
    parser.add_argument("--hot-samples", type=int, default=0)
    parser.add_argument("--large-group-clients", type=int, default=0)
    parser.add_argument("--keep-pools", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=(
            "default",
            "lora",
            "kv",
            "work",
            "work-no-adapter",
            "domains",
            "kv-cohort",
            "work-cohort",
            "domains-cohort",
            "balanced",
            "balanced-cohort",
            "stable",
            "stable-cohort",
            "capacity",
            "capacity-cohort",
            "capacity-random",
            "capacity-random-cohort",
            *SIMPLE_POLICIES,
            *(p + "-cohort" for p in SIMPLE_POLICIES),
        ),
    )
    args = parser.parse_args()
    pools = json.loads(args.pools.read_text())
    if args.arms:
        pools = [
            next(
                p
                for p in pools
                if p["arm"]
                == (
                    "work"
                    if arm.removesuffix("-cohort")
                    in {*SIMPLE_POLICIES, "capacity", "capacity-random"}
                    else "domains"
                    if arm.startswith(("balanced", "stable"))
                    else arm.removesuffix("-cohort")
                )
            )
            | {"arm": arm}
            for arm in args.arms
        ]
    if (
        any(
            (
                p["arm"].startswith("work")
                or p["arm"].startswith(("domains", "balanced", "stable"))
                or p["arm"].removesuffix("-cohort") in SIMPLE_POLICIES
                or p["arm"].startswith("capacity")
            )
            for p in pools
        )
        and args.work_rates is None
    ):
        parser.error("measured-work policies require --work-rates from calibration")
    clients = json.loads(args.clients.read_text())
    if args.vary_placement:
        random.Random(args.seed).shuffle(clients)
        for index, source in enumerate(clients):
            source["original_client_index"] = source["client_index"]
            source["client_index"] = index
    for source in clients:
        source["prompt_seed"] = args.seed
        source["vocabulary"] = sorted(
            {token for prompt in source["prompts"] for token in prompt}
        )
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
            seed=args.seed,
            groups=args.groups,
            dialogue_turns=args.dialogue_turns,
            samples=args.samples,
            rounds=args.rounds,
            stagger_seconds=args.stagger_seconds,
            hot_samples=args.hot_samples,
            large_group_clients=args.large_group_clients,
            decode_tokens=args.decode_tokens,
            long_prefix=args.long_prefix,
            warm_prefixes=args.warm_prefixes,
            repeat_prefixes=args.repeat_prefixes,
            version_turnover=not args.fixed_version,
            cold_versions=args.cold_versions,
            separate_cohort_transport=True,
            generation_transport_scope="sample_task",
            preparation_http_version="HTTP/2" if args.http2 else "HTTP/1.1",
            preparation_transport_scope="sample_task",
            preparation_request_timeout=args.request_timeout,
            duration_seconds=args.duration_seconds,
            work_rates=json.loads(args.work_rates.read_text())
            if args.work_rates
            else None,
        )
        per_round = len(clients) * args.groups * args.samples
        if args.hot_samples:
            per_round += args.hot_samples - args.groups * args.samples
        config["expected_requests"] = per_round * args.rounds * args.dialogue_turns
        config["max_concurrency"] = per_round
    config["adapter_count"] = sum(len(c["versions"]) for c in clients)
    if args.vary_placement:
        config["affinity_salt"] = f"trial-{args.seed}:"
        config["vary_placement"] = True
    config["arrival_rate"] = args.arrival_rate
    config["request_timeout"] = args.request_timeout
    if args.http2:
        config["http2"] = True
        if args.workload != "dapo":
            config["http2_client_shards"] = 8
    root = Path(__file__).resolve().parents[3]
    config["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "src/spindle/inference/lora_routing.py",
            "src/spindle/inference/routing_work.py",
            "src/spindle/inference/routing_domains.py",
            "src/spindle/inference/routing_simple.py",
            "src/spindle/inference/routing_capacity.py",
            "src/spindle/inference/sampling.py",
            "src/spindle/providers/modal/lora_routing.py",
            "scripts/benchmarks/lora_routing/cache_replay.py",
            "scripts/benchmarks/lora_routing/cohort_prefill.py",
            "scripts/benchmarks/lora_routing/h2_diagnostics.py",
        )
    }
    if args.work_rates:
        config["work_rates"] = json.loads(args.work_rates.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in config["source_sha256"].items():
        content = (root / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def run(executions):
        async def one(pool):
            result = await run_arm.remote.aio(
                pool["pool"], pool["arm"], clients, config, executions
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
        with modal.enable_output(), app.run(), modal.Dict.ephemeral() as executions:
            results = asyncio.run(run(executions))
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
