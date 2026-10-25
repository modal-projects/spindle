"""Fully asynchronous 128 x 8 GRPO with native score-centered MIS.

Rollout producers continue throughout backward, optimizer and publication.
Completed groups are consumed in completion order, without a staleness cutoff.
"""

import argparse
import asyncio
import gzip
import json
import math
import os
import pickle
import random
import time
from contextlib import ExitStack
from pathlib import Path

import httpx
import modal
import tinker
from tinker import types

import spindle
from spindle.deployments import DeploymentConfig
from spindle.providers.modal.deployment_apps import build_rollout_app
from spindle.providers.modal.fft_pool import FFTPoolSpec

from .config import (
    CHECKPOINT_INTERVAL,
    DATASET,
    DATASET_REVISION,
    GROUPS,
    LEARNING_RATE,
    MIS_HIGH,
    MIS_LOW,
    PROMPTS_PER_UPDATE,
    SEED,
    TEMPERATURE,
    TOP_P,
    TRAINING_STEPS,
    TRAJECTORIES_PER_PROMPT,
    config,
    sampler_config,
)
from .data import load_tasks
from .workers import app as worker_app
from .workers import measure_capacity, sample_trajectory


def tensor(values, dtype="float32"):
    return types.TensorData(data=values, dtype=dtype, shape=[len(values)])


def datum(row, advantage, batch_tokens):
    # Metadata is indexed by target token, so remove the first prompt token.
    return types.Datum(
        model_input=types.ModelInput.from_ints(row["tokens"][:-1]),
        loss_fn_inputs={
            "target_tokens": tensor(row["tokens"][1:], "int64"),
            "weights": tensor([active / batch_tokens for active in row["active"][1:]]),
            "advantages": tensor([advantage * active for active in row["active"][1:]]),
            "logprobs": tensor(row["logq"][1:]),
            "sampling_mask_ids": tensor(row["support"], "int64"),
            "sampling_mask_offsets": tensor(row["offsets"][1:], "int64"),
            "sampling_mask_logprobs": tensor(row["support_logq"]),
        },
    )


def read_row(path):
    with gzip.open(path, "rb") as source:
        return pickle.load(source)


def trajectory_metadata(row):
    return {
        key: row[key]
        for key in (
            "reward",
            "format_error",
            "generated_tokens",
            "versions",
            "group",
            "started_at",
            "finished_at",
            "stop_reason",
            "turns",
        )
    }


DIAGNOSTICS = ("entropy", "sampler_trainer_kl", "mis_masked_fraction")


def diagnostic_means(metrics):
    # Tinker combines :mean by datum count, which biases token means when it
    # splits a large request. Additive totals survive that SDK reduction.
    result = dict(metrics)
    exact = all(f"{name}:sum" in metrics for name in DIAGNOSTICS)
    if exact:
        for name in DIAGNOSTICS:
            result[f"{name}:mean"] = metrics[f"{name}:sum"] / max(
                metrics["tokens:sum"], 1.0
            )
    return result, "token_mean" if exact else "sdk_chunk_means"


def initial_rollout_cache(path, tasks):
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {
        "model": config.model,
        "model_revision": config.model_revision,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "groups": GROUPS,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "batch": [PROMPTS_PER_UPDATE, TRAJECTORIES_PER_PROMPT],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("initial rollout cache uses a different recipe")
    if manifest["arguments"]["checkpoint"] or manifest["arguments"]["start_step"]:
        raise ValueError("initial rollout cache must start from the pretrained model")
    events = [
        json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()
    ]
    if any(event["event"] == "training_completed" for event in events):
        raise ValueError("initial rollout cache must precede any completed update")
    order = list(range(len(tasks)))
    random.Random(SEED).shuffle(order)
    prompts = {e["prompt"]: e for e in events if e["event"] == "prompt_started"}
    for number, prompt in prompts.items():
        if prompt["instance_id"] != tasks[order[number % len(order)]]["instance_id"]:
            raise ValueError("initial rollout cache has a different task order")
    completed = {
        (e["prompt"], e["index"])
        for e in events
        if e["event"] == "trajectory_completed"
    }
    capacity = manifest["capacity"]
    if set(capacity) != {g["name"] for g in GROUPS} or any(
        not math.isfinite(c["tokens_per_second"]) or c["tokens_per_second"] <= 0
        for c in capacity.values()
    ):
        raise ValueError("initial rollout cache has invalid capacity measurements")
    return {
        "path": path,
        "prompts": prompts,
        "completed": completed,
        "capacity": capacity,
    }


def wait_for_sampler_evidence(paths):
    deadline = time.monotonic() + 1800
    for path in paths:
        while True:
            try:
                reports = json.loads(path.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"sampler validation did not finish: {path}")
            time.sleep(5)
        for report in reports:
            if (
                report.get("error")
                or not report["weight_validation"]["baseline_weights_match"]
            ):
                raise RuntimeError(
                    f"sampler validation failed: {path}: {report.get('error')}"
                )
        print(
            json.dumps({"event": "sampler_validation_passed", "path": str(path)}),
            flush=True,
        )


async def train(training, gateways, weights, args, tasks, *, cache=None):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events_file = (output / "events.jsonl").open("a", buffering=1)

    def event(kind, **values):
        row = {"event": kind, "at": time.time(), **values}
        events_file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    version = 1
    completed = asyncio.Queue(maxsize=args.inflight_prompts)
    rng = random.Random(SEED)
    order = list(range(len(tasks)))
    rng.shuffle(order)
    next_task = 0
    sequence = 0
    groups = list(gateways)

    async def producer():
        nonlocal next_task, sequence
        while True:
            task = tasks[order[next_task % len(order)]]
            next_task += 1
            group_number = sequence
            sequence += 1
            requested_version = version
            assignments = rng.choices(
                groups, weights=weights, k=TRAJECTORIES_PER_PROMPT
            )
            if cache is not None and group_number in cache["prompts"]:
                assignments = cache["prompts"][group_number]["groups"]
            event(
                "prompt_started",
                prompt=group_number,
                instance_id=task["instance_id"],
                version=requested_version,
                groups=assignments,
            )

            async def one(
                index,
                group,
                task=task,
                group_number=group_number,
                requested_version=requested_version,
            ):
                path = output / "trajectories" / f"{group_number:08d}-{index}.pkl.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if cache is not None and (group_number, index) in cache["completed"]:
                    source = cache["path"] / "trajectories" / path.name
                    row = await asyncio.to_thread(read_row, source)
                    if (
                        row["instance_id"] != task["instance_id"]
                        or row["group"] != group
                        or not row["versions"]
                        or any(
                            v["start"] != 1 or v["end"] != 1 for v in row["versions"]
                        )
                    ):
                        raise ValueError(
                            "cached trajectory is not from the initial policy"
                        )
                    path.write_bytes(source.read_bytes())
                    metadata = trajectory_metadata(row)
                    event(
                        "trajectory_restored",
                        prompt=group_number,
                        index=index,
                        source=str(source),
                        **metadata,
                    )
                    return str(path), metadata
                for attempt in range(2):
                    try:
                        event(
                            "trajectory_started",
                            prompt=group_number,
                            index=index,
                            group=group,
                            version=requested_version,
                            attempt=attempt,
                        )
                        compressed = await sample_trajectory.remote.aio(
                            task,
                            gateways[group],
                            group,
                            training.model_id,
                            requested_version,
                            SEED + group_number * 10000 + index * 1000,
                        )
                        path.write_bytes(compressed)
                        row = pickle.loads(gzip.decompress(compressed))
                        metadata = trajectory_metadata(row)
                        event(
                            "trajectory_completed",
                            prompt=group_number,
                            index=index,
                            **metadata,
                        )
                        return str(path), metadata
                    except Exception as exc:
                        event(
                            "trajectory_failed",
                            prompt=group_number,
                            index=index,
                            attempt=attempt,
                            error=repr(exc),
                        )
                        if attempt:
                            raise

            try:
                rows = await asyncio.gather(
                    *(one(i, g) for i, g in enumerate(assignments))
                )
                await completed.put((group_number, rows))
            except Exception as exc:  # noqa: BLE001 - transfer the failure to the trainer task
                await completed.put(exc)
                return

    producers = [asyncio.create_task(producer()) for _ in range(args.inflight_prompts)]
    try:
        for step in range(args.start_step + 1, args.start_step + args.steps + 1):
            batch = []
            for _ in range(PROMPTS_PER_UPDATE):
                item = await asyncio.wait_for(completed.get(), timeout=7200)
                if isinstance(item, Exception):
                    raise item
                batch.append(item)
            started = time.time()
            total_tokens = sum(
                m["generated_tokens"] for _, rows in batch for _, m in rows
            )
            if not total_tokens:
                raise RuntimeError("training batch has no generated tokens")
            event(
                "training_started",
                step=step,
                version=version,
                prompts=[i for i, _ in batch],
                tokens=total_tokens,
            )
            metrics = {}
            diagnostic_aggregation = set()
            zero_variance_prompts = 0
            for prompt_number, rows in batch:
                rewards = [m["reward"] for _, m in rows]
                mean = sum(rewards) / len(rewards)
                std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
                zero_variance_prompts += int(std == 0)
                data = []
                for (path, metadata), reward in zip(rows, rewards, strict=True):
                    row = await asyncio.to_thread(read_row, path)
                    advantage = (reward - mean) / (std + 1e-6)
                    data.append(datum(row, advantage, total_tokens))
                backward_started = time.time()
                result = await asyncio.wait_for(
                    training.forward_backward(
                        data,
                        "score_centered_mis",
                        {
                            "mis_low": MIS_LOW,
                            "mis_high": MIS_HIGH,
                            "sampling_temperature": TEMPERATURE,
                        },
                    ).result_async(),
                    timeout=7200,
                )
                group_tokens = sum(m["generated_tokens"] for _, m in rows)
                reported_metrics, aggregation = diagnostic_means(result.metrics)
                diagnostic_aggregation.add(aggregation)
                event(
                    "backward_completed",
                    step=step,
                    prompt=prompt_number,
                    elapsed=time.time() - backward_started,
                    tokens=group_tokens,
                    metrics=reported_metrics,
                    diagnostic_aggregation=aggregation,
                )
                for name, value in reported_metrics.items():
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"non-finite training metric {name}: {value}"
                        )
                    if name.endswith(":mean"):
                        metrics[name] = (
                            metrics.get(name, 0.0) + value * group_tokens / total_tokens
                        )
            optimizer = await asyncio.wait_for(
                training.optim_step(
                    types.AdamParams(
                        learning_rate=LEARNING_RATE,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                        weight_decay=0.0,
                        grad_clip_norm=1.0,
                    )
                ).result_async(),
                timeout=7200,
            )
            if any(not math.isfinite(v) for v in optimizer.metrics.values()):
                raise RuntimeError(f"non-finite optimizer metrics: {optimizer.metrics}")
            if optimizer.metrics["update_successful:mean"] != 1:
                raise RuntimeError(f"optimizer skipped an update: {optimizer.metrics}")
            publication = await asyncio.wait_for(
                training.save_weights_for_sampler(f"step-{step:05d}").result_async(),
                timeout=3600,
            )
            version += 1
            all_meta = [m for _, rows in batch for _, m in rows]
            group_metrics = {}
            for group in gateways:
                selected = [m for m in all_meta if m["group"] == group]
                if not selected:
                    continue
                ages = [
                    version - 1 - min(v["start"] for v in m["versions"])
                    for m in selected
                ]
                group_metrics[group] = {
                    "trajectories": len(selected),
                    "reward": sum(m["reward"] for m in selected) / len(selected),
                    "format_error_rate": sum(m["format_error"] for m in selected)
                    / len(selected),
                    "policy_age_mean": sum(ages) / len(ages),
                    "policy_age_max": max(ages),
                }
            event(
                "training_completed",
                step=step,
                version=version,
                publication=publication.path,
                elapsed=time.time() - started,
                metrics=metrics,
                diagnostic_aggregation=sorted(diagnostic_aggregation),
                optimizer=optimizer.metrics,
                reward=sum(m["reward"] for m in all_meta) / len(all_meta),
                format_error_rate=sum(m["format_error"] for m in all_meta)
                / len(all_meta),
                mean_generated_tokens=total_tokens / len(all_meta),
                queued_prompts=completed.qsize(),
                zero_variance_prompts=zero_variance_prompts,
                groups=group_metrics,
            )
            if (
                step == args.start_step + 1
                or step % CHECKPOINT_INTERVAL == 0
                or step == args.start_step + args.steps
            ):
                checkpoint = await asyncio.wait_for(
                    training.save_state(f"step-{step:05d}").result_async(),
                    timeout=3600,
                )
                (output / "checkpoint.json").write_text(
                    json.dumps({"step": step, "path": checkpoint.path})
                )
                event("checkpoint", step=step, path=checkpoint.path)
    finally:
        for producer in producers:
            producer.cancel()
        await asyncio.gather(*producers, return_exceptions=True)
        events_file.close()


async def run_with_sampler_validation(training, validation=None):
    async with asyncio.TaskGroup() as group:
        if validation is not None:
            group.create_task(validation)
        group.create_task(training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--inflight-prompts", type=int, default=192)
    parser.add_argument("--checkpoint")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--sampler-evidence", type=Path, nargs="+", default=[])
    parser.add_argument("--initial-rollout-cache", type=Path)
    args = parser.parse_args()
    tasks = load_tasks()
    if args.initial_rollout_cache and (args.checkpoint or args.start_step):
        parser.error("initial rollout reuse requires a fresh pretrained trainer")
    cache = (
        initial_rollout_cache(args.initial_rollout_cache, tasks)
        if args.initial_rollout_cache
        else None
    )
    url = modal.Function.from_name(
        config.platform["frontend"], "server", environment_name="kailash-dev"
    ).get_web_url()
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ["TINKER_API_KEY"]
    )
    service = tinker.ServiceClient(base_url=url, api_key=key)
    print(json.dumps({"event": "creating_trainer"}), flush=True)
    training = spindle.create_full_training_client(service, config.model)
    print(
        json.dumps({"event": "trainer_created", "model_id": training.model_id}),
        flush=True,
    )
    try:
        if args.checkpoint:
            training.load_state_with_optimizer(args.checkpoint).result(timeout=7200)
        publication = training.save_weights_for_sampler("initial").result(timeout=3600)
        print(
            json.dumps({"event": "initial_publication", "path": publication.path}),
            flush=True,
        )
        wait_for_sampler_evidence(args.sampler_evidence)
        gateways = {}
        with modal.enable_output(), ExitStack() as stack:
            stack.enter_context(worker_app.run(environment_name="kailash-dev"))
            for group in GROUPS:
                deployment = DeploymentConfig.create(sampler_config(group))
                pool = FFTPoolSpec(deployment.definition_id, training.model_id, True, 0)
                app, server = build_rollout_app(deployment, pool)
                stack.enter_context(app.run(environment_name="kailash-dev"))
                gateways[group["name"]] = server.get_url()

            async def capacities():
                return await asyncio.gather(
                    *(
                        measure_capacity.remote.aio(url, training.model_id, 1)
                        for url in gateways.values()
                    )
                )

            measured = (
                [cache["capacity"][group] for group in gateways]
                if cache is not None
                else asyncio.run(capacities())
            )
            weights = [m["tokens_per_second"] for m in measured]
            manifest = {
                "model": config.model,
                "model_revision": config.model_revision,
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "groups": GROUPS,
                "capacity": dict(zip(gateways, measured, strict=True)),
                "capacity_source": str(cache["path"]) if cache is not None else None,
                "compute_regions": {
                    group["name"]: sampler_config(group).inference_region
                    if sampler_config(group).inference_region is not None
                    else config.platform["modal"]["region"]
                    for group in GROUPS
                },
                "top_p": TOP_P,
                "temperature": TEMPERATURE,
                "lr": LEARNING_RATE,
                "batch": [PROMPTS_PER_UPDATE, TRAJECTORIES_PER_PROMPT],
                "model_id": training.model_id,
                "initial_publication": publication.path,
                "arguments": {
                    k: [str(p) for p in v]
                    if k == "sampler_evidence"
                    else str(v)
                    if isinstance(v, Path)
                    else v
                    for k, v in vars(args).items()
                },
            }
            Path(args.output).mkdir(parents=True, exist_ok=True)
            (Path(args.output) / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            async def validate_samplers():
                measured = await capacities()
                (Path(args.output) / "startup-validation.json").write_text(
                    json.dumps(dict(zip(gateways, measured, strict=True)), indent=2)
                )

            asyncio.run(
                run_with_sampler_validation(
                    train(training, gateways, weights, args, tasks, cache=cache),
                    validate_samplers() if cache is not None else None,
                )
            )
    finally:
        try:
            response = httpx.post(
                f"{url}/api/v1/unload_model",
                headers={"X-API-Key": key},
                json={"model_id": training.model_id},
                timeout=300,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(
                json.dumps({"event": "trainer_cleanup_failed", "error": str(exc)}),
                flush=True,
            )


if __name__ == "__main__":
    main()
