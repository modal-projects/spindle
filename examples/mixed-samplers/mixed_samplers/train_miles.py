"""Independent asynchronous Miles LoRA clients over eight shared sampler groups."""

import argparse
import asyncio
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import httpx
import modal
import tinker

from spindle.deployments import DeploymentConfig
from spindle.providers.modal.deployment_apps import build_rollout_app
from spindle.providers.modal.lora_pool import LoraPoolSpec

from .config import (
    DATASET,
    DATASET_REVISION,
    GROUPS,
    LEARNING_RATE,
    PROMPTS_PER_UPDATE,
    SEED,
    TEMPERATURE,
    TOP_P,
    TRAJECTORIES_PER_PROMPT,
)
from .data import load_tasks
from .miles_config import config, sampler_config
from .miles_workers import app as worker_app
from .miles_workers import measure_capacity, sample_trajectory
from .publication import latest_ref
from .train import read_initial_rollouts, train


def sampler_groups():
    groups = []
    for group in GROUPS:
        settings = sampler_config(group).sglang_cfg
        groups.append(
            {
                **group,
                **{
                    key: settings[key]
                    for key in (
                        "moe_runner_backend",
                        "lora_backend",
                        "kv_cache_dtype",
                        "tp_size",
                        "ep_size",
                    )
                },
            }
        )
    return groups


def previous_manifest(path, client_count):
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {
        "backend": "miles",
        "parameterization": "lora",
        "model": config.model,
        "model_revision": config.model_revision,
        "rank": 32,
        "alpha": 32,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "batch_per_client": [PROMPTS_PER_UPDATE, TRAJECTORIES_PER_PROMPT],
        "clients": client_count,
        "client_seeds": [SEED + index * 1_000_000 for index in range(client_count)],
    }
    # Older manifests recorded the shared group definitions before Miles'
    # LoRA runner overrides. Their executed sampler settings were the same.
    if manifest.get("groups") not in (GROUPS, sampler_groups()) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("previous Miles run uses a different recipe or client seed")
    capacity = manifest["capacity"]
    if set(capacity) != {g["name"] for g in GROUPS} or any(
        not math.isfinite(row["tokens_per_second"]) or row["tokens_per_second"] <= 0
        for row in capacity.values()
    ):
        raise ValueError("previous Miles run has invalid capacity measurements")
    return manifest


def initial_client_rollouts(path, tasks, client_count):
    manifest = previous_manifest(path, client_count)
    if (
        manifest.get("initial_versions") != [1] * client_count
        or manifest.get("initial_steps", [0] * client_count) != [0] * client_count
    ):
        raise ValueError("initial Miles rollouts must come from untrained adapters")
    return {
        "capacity": manifest["capacity"],
        "clients": [
            read_initial_rollouts(
                path / f"client-{index}",
                tasks,
                client_seed=SEED + index * 1_000_000,
            )
            for index in range(client_count)
        ],
    }


def checkpoint_resume(path, client_count, target_steps):
    manifest = previous_manifest(path, client_count)
    if manifest.get("lr") != LEARNING_RATE or manifest.get("context_length") != (
        config.max_context_length
    ):
        raise ValueError("previous Miles run uses different training settings")
    checkpoints = [
        json.loads((path / f"client-{index}" / "checkpoint.json").read_text())
        for index in range(client_count)
    ]
    for checkpoint in checkpoints:
        step = checkpoint.get("step")
        if type(step) is not int or not 0 < step <= target_steps:
            raise ValueError("checkpoint step must be positive and within --steps")
        if not isinstance(checkpoint.get("path"), str) or not checkpoint[
            "path"
        ].startswith("tinker://"):
            raise ValueError("checkpoint must reference a saved Tinker optimizer state")
    if all(checkpoint["step"] == target_steps for checkpoint in checkpoints):
        raise ValueError("all clients already reached the requested --steps")
    return {"capacity": manifest["capacity"], "checkpoints": checkpoints}


async def run_clients(
    args, service, gateways, tasks, url, key, *, cache=None, resume=None
):
    clients = []
    initial_versions = []
    initial_steps = (
        [checkpoint["step"] for checkpoint in resume["checkpoints"]]
        if resume is not None
        else [0] * args.clients
    )
    args.output.mkdir(parents=True, exist_ok=True)
    events = args.output / "clients.jsonl"

    def event(kind, **values):
        record = {"event": kind, "at": time.time(), **values}
        with events.open("a") as out:
            out.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    try:
        if args.trainer_validation:
            deadline = time.monotonic() + 3600
            while not args.trainer_validation.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Miles trainer validation did not finish")
                await asyncio.sleep(5)
            if not json.loads(args.trainer_validation.read_text())["passed"]:
                raise RuntimeError("Miles trainer validation failed")
        if args.validation_adapters:
            validated = json.loads(args.validation_adapters.read_text())
            if not validated["passed"]:
                raise ValueError("adapter validation did not pass")
            checks = []
            for model_id in validated["model_ids"]:
                ref = await latest_ref(model_id, config.platform["storage"]["bulletin"])
                results = await asyncio.gather(
                    *(
                        measure_capacity.remote.aio(gateway, model_id, ref.version)
                        for gateway in gateways.values()
                    )
                )
                checks.append(
                    {
                        "model_id": model_id,
                        "version": ref.version,
                        "groups": dict(zip(gateways, results, strict=True)),
                    }
                )
                event(
                    "trained_adapter_sampler_checks_passed",
                    model_id=model_id,
                    version=ref.version,
                )
            (args.output / "sampler-validation.json").write_text(
                json.dumps({"passed": True, "trained_adapters": checks}, indent=2)
            )
        # Startup is shared. Once capacity is measured, each client owns its own
        # producer queue, update counter, publication and checkpoint sequence.
        for index in range(args.clients):
            event("creating_client", client=index)
            training = await asyncio.wait_for(
                service.create_lora_training_client_async(
                    config.model,
                    rank=32,
                    train_attn=True,
                    train_mlp=True,
                    train_unembed=False,
                ),
                3600,
            )
            clients.append(training)
            if resume is not None:
                checkpoint = resume["checkpoints"][index]
                await asyncio.wait_for(
                    training.load_state_with_optimizer(
                        checkpoint["path"]
                    ).result_async(),
                    7200,
                )
                directory = args.output / f"client-{index}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "checkpoint.json").write_text(json.dumps(checkpoint))
                event("client_resumed", client=index, **checkpoint)
            publication = await asyncio.wait_for(
                training.save_weights_for_sampler("initial").result_async(), 1800
            )
            ref = await latest_ref(
                training.model_id, config.platform["storage"]["bulletin"]
            )
            initial_versions.append(ref.version)
            event(
                "client_ready",
                client=index,
                model_id=training.model_id,
                publication=publication.path,
                version=ref.version,
            )
        measured = await asyncio.gather(
            *(
                measure_capacity.remote.aio(
                    gateway, clients[0].model_id, initial_versions[0]
                )
                for gateway in gateways.values()
            )
        )
        previous = cache if cache is not None else resume
        capacity = [previous["capacity"][g] for g in gateways] if previous else measured
        weights = [row["tokens_per_second"] for row in capacity]
        manifest = {
            "backend": "miles",
            "parameterization": "lora",
            "rank": 32,
            "alpha": 32,
            "model": config.model,
            "model_revision": config.model_revision,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "groups": sampler_groups(),
            "capacity": dict(zip(gateways, capacity, strict=True)),
            "startup_capacity": dict(zip(gateways, measured, strict=True)),
            "initial_rollout_cache": str(args.initial_rollout_cache)
            if args.initial_rollout_cache
            else None,
            "batch_per_client": [PROMPTS_PER_UPDATE, TRAJECTORIES_PER_PROMPT],
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
            "lr": LEARNING_RATE,
            "model_ids": [client.model_id for client in clients],
            "initial_versions": initial_versions,
            "initial_steps": initial_steps,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "policy_age_unit": "optimizer_updates",
            "clients": args.clients,
            "rollout_concurrency_per_client": args.rollout_concurrency,
            "client_seeds": [SEED + index * 1_000_000 for index in range(args.clients)],
            "stagger_seconds": args.stagger_seconds,
            "steps_per_client": args.steps,
            "sampler_parameterization": "quantized frozen base plus BF16 per-client LoRA adapters",
            "context_length": config.max_context_length,
            "scheduling": "independent completion-order producers and updates, no policy-age limit",
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

        async def one(index, client):
            async def version_resolver():
                ref = await latest_ref(
                    client.model_id, config.platform["storage"]["bulletin"]
                )
                return ref.version

            await asyncio.sleep(index * args.stagger_seconds)
            event("client_training_started", client=index, model_id=client.model_id)
            client_args = SimpleNamespace(
                output=args.output / f"client-{index}",
                start_step=initial_steps[index],
                steps=args.steps - initial_steps[index],
                inflight_prompts=args.inflight_prompts,
            )
            await train(
                client,
                gateways,
                weights,
                client_args,
                tasks,
                cache=cache["clients"][index] if cache is not None else None,
                rollout_worker=sample_trajectory,
                client_seed=SEED + index * 1_000_000,
                initial_version=initial_versions[index],
                version_resolver=version_resolver,
                rollout_concurrency=args.rollout_concurrency,
            )
            event("client_training_completed", client=index, model_id=client.model_id)

        # gather waits for termination only; it introduces no per-step barrier.
        running = [asyncio.create_task(one(i, c)) for i, c in enumerate(clients)]
        try:
            await asyncio.gather(*running)
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
    finally:
        async with httpx.AsyncClient(timeout=300) as http:
            for client in clients:
                response = await http.post(
                    url + "/api/v1/unload_model",
                    headers={"X-API-Key": key},
                    json={"model_id": client.model_id},
                )
                response.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--stagger-seconds", type=float, default=60.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Target total optimizer updates per client",
    )
    parser.add_argument("--inflight-prompts", type=int, default=192)
    parser.add_argument("--rollout-concurrency", type=int, default=512)
    parser.add_argument("--trainer-validation", type=Path)
    parser.add_argument("--validation-adapters", type=Path)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--initial-rollout-cache", type=Path)
    recovery.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if not 1 <= args.clients <= config.trainer_max_clients_per_instance:
        raise ValueError("clients must fit the configured shared Miles trainer")
    if args.rollout_concurrency < 1:
        raise ValueError("rollout concurrency must be positive")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    tasks = load_tasks()
    cache = (
        initial_client_rollouts(args.initial_rollout_cache, tasks, args.clients)
        if args.initial_rollout_cache
        else None
    )
    resume = (
        checkpoint_resume(args.resume_from, args.clients, args.steps)
        if args.resume_from
        else None
    )
    url = modal.Function.from_name(
        config.platform["frontend"], "server", environment_name="kailash-dev"
    ).get_web_url()
    key = args.api_key_file.read_text().strip()
    service = tinker.ServiceClient(base_url=url, api_key=key)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "session.json").write_text(
        json.dumps({"session_id": service.holder.get_session_id()})
    )
    gateways = {}
    with modal.enable_output(), ExitStack() as stack:
        stack.enter_context(worker_app.run(environment_name="kailash-dev"))
        for group in GROUPS:
            deployment = DeploymentConfig.create(sampler_config(group))
            app, server = build_rollout_app(
                deployment, LoraPoolSpec(deployment.definition_id)
            )
            stack.enter_context(app.run(environment_name="kailash-dev"))
            gateways[group["name"]] = server.get_url()
        asyncio.run(
            run_clients(
                args, service, gateways, tasks, url, key, cache=cache, resume=resume
            )
        )


if __name__ == "__main__":
    main()
