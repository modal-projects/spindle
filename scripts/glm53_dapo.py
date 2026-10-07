"""Independent GLM LoRA clients using Spindle's Engine command scheduler."""

import asyncio
import json
import math
import time
from pathlib import Path

import modal
import numpy as np
import torch
from tinker import AdamParams, LoraConfig
from tinker.types.forward_backward_input import ForwardBackwardInput

from glm53_dapo_data import (
    MODEL,
    request_sample,
    sample_problems,
    save_rollouts,
    summary,
    training_data,
    write_json,
)
from spindle.backends.contract import ModelSpec
from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_lora import MilesCommandBackend
from spindle.engine import Engine, FutureStatus, OperationKind
from spindle.engine.operations import (
    SaveCheckpointPayload,
    SaveWeightsForSamplerPayload,
)
from spindle.engine.spmd import DistributedExecutor


async def run_clients(engine, sampler, config, dataset, root, volume):
    report = {"config": config, "client_steps": [], "events": [], "evaluations": []}
    states = [{} for _ in range(config["clients"])]
    checkpoints = [None] * config["clients"]
    save_lock = asyncio.Lock()
    preparation_slots = asyncio.Semaphore(config.get("prepared_batches", 2))
    deadline = time.monotonic() + 19 * 3600

    async def record(client, phase, step, **details):
        row = {
            "client": client,
            "phase": phase,
            "step": step,
            "time": time.time(),
            **details,
        }
        report["events"].append(row)
        lane = "rollout" if phase in {"sampling", "rollout_ready"} else "update"
        row["lane"] = lane
        states[client][lane] = row
        print("GLM CLIENT", json.dumps(row), flush=True)
        async with save_lock:
            write_json(root / "status.json", {"phase": "running", "clients": states})
            write_json(root / "report.json", report)
        return row["time"]

    async def client_loop(client):
        model = f"{config['run_id']}-client{client}"
        sequence = 0

        async def submit(kind, payload):
            nonlocal sequence
            sequence += 1
            # The same queue used by Engine HTTP ingress; keep large route tensors
            # in memory rather than encoding a second local HTTP request.
            request = await engine._submit(kind, model, sequence, payload)
            while True:
                state = await engine.retrieve_future(request, timeout=30)
                if state is None:
                    raise RuntimeError(f"Engine lost {request}")
                if state.status == FutureStatus.FAILED:
                    raise RuntimeError(state.error)
                if state.status == FutureStatus.COMPLETE:
                    return state.result

        await engine.accept_model(
            model,
            ModelSpec(
                base_model=MODEL,
                parameterization="lora",
                lora_config=LoraConfig(rank=config["lora_rank"], train_unembed=False),
            ),
        )
        await asyncio.sleep(client * config["stagger_s"])
        cursor = 0
        local_config = {
            **config,
            "concurrency": max(1, config["concurrency"] // config["clients"]),
        }

        async def collect(step, version):
            start_time = await record(client, "sampling", step, policy_version=version)
            offset = (step - 1) * config["groups"]
            order = dataset["orders"][client][offset : offset + config["groups"]]
            if len(order) != config["groups"]:
                raise RuntimeError("Training dataset exhausted")
            groups = await asyncio.to_thread(
                sample_problems,
                sampler,
                [(client, dataset["train"][index]) for index in order],
                local_config,
                version,
            )
            ready_time = time.time()
            await record(client, "rollout_ready", step, policy_version=version)
            return groups, version, start_time, ready_time

        pending = asyncio.create_task(collect(1, 0))
        saving = None
        previous_update_end = None
        try:
            for step in range(1, config["steps"] + 1):
                groups, behavior_version, start_time, rollout_end = await pending
                pending = None
                # Exactly one batch ahead. Training update s consumes version
                # s-2 after the first update; PPO uses its recorded behavior logps.
                if step < config["steps"] and time.monotonic() < deadline:
                    pending = asyncio.create_task(collect(step + 1, step - 1))
                sampling_s = rollout_end - start_time
                cursor += len(groups)
                policy_lag = step - 1 - behavior_version
                assert 0 <= policy_lag <= 1, "Rollout exceeded one-update policy lag"
                await asyncio.sleep(0)  # Admit the next rollout before training.
                if config.get("functional_validation"):
                    # Exercise nonzero gradients even when short answers truncate.
                    for group in groups:
                        for member, sample in enumerate(group["samples"]):
                            sample["truncated"] = False
                            sample["reward"] = 1.0 if member % 2 else -1.0
                saving = asyncio.create_task(
                    asyncio.to_thread(
                        save_rollouts,
                        root / f"client{client}-train-{step:04d}.jsonl.gz",
                        groups,
                    )
                )
                async with preparation_slots:
                    data, old_logprobs, trained_tokens = await asyncio.to_thread(
                        training_data,
                        groups,
                        routing_replay=True,
                    )
                    await record(
                        client,
                        "training",
                        step,
                        tokens=trained_tokens,
                        policy_lag=policy_lag,
                        behavior_policy_version=behavior_version,
                    )
                    t0 = time.monotonic()
                    output = await submit(
                        OperationKind.FORWARD_BACKWARD,
                        ForwardBackwardInput(
                            data=data,
                            loss_fn="ppo",
                            loss_fn_config={
                                "clip_low_threshold": config["clip_low"],
                                "clip_high_threshold": config["clip_high"],
                            },
                        ),
                    )
                    forward_s = time.monotonic() - t0
                    del data
                differences = np.concatenate(
                    [
                        np.asarray(result["logprobs"]["data"][prefix:])
                        - np.asarray(old)
                        for result, (prefix, old) in zip(
                            output["loss_fn_outputs"], old_logprobs, strict=True
                        )
                    ]
                )
                assert np.isfinite(differences).all()
                if policy_lag == 0:
                    assert np.abs(differences).mean() < 0.5, (
                        "Trainer/sampler logprobs disagree for the same policy"
                    )
                t0 = time.monotonic()
                optimizer = await submit(
                    OperationKind.OPTIM_STEP,
                    AdamParams(
                        learning_rate=config["learning_rate"],
                        beta1=config["beta1"],
                        beta2=config["beta2"],
                        weight_decay=config["weight_decay"],
                        grad_clip_norm=1.0,
                    ),
                )
                optimizer_s = time.monotonic() - t0
                metrics = optimizer["metrics"]
                assert metrics["update_successful:mean"] == 1
                assert math.isfinite(metrics["grad_norm:mean"])
                if config.get("functional_validation"):
                    assert metrics["grad_norm:mean"] > 0
                await record(client, "publishing", step)
                t0 = time.monotonic()
                await submit(
                    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
                    SaveWeightsForSamplerPayload(publish_version=step),
                )
                probe = await asyncio.to_thread(
                    request_sample,
                    sampler[client % len(sampler)],
                    {
                        "input_ids": dataset["eval"][0]["tokens"],
                        "weight_run_id": model,
                        "weight_version": {"exact_version": step},
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    },
                )
                assert probe["meta_info"]["weight_version_start"] == step
                publication_s = time.monotonic() - t0
                update_end = time.time()
                # Persistence runs alongside sampling/training, with at most one
                # save per client retained before consuming the next batch.
                await saving
                saving = None
                row = {
                    "client": client,
                    "step": step,
                    "cursor": cursor,
                    "start_time": start_time,
                    "end_time": update_end,
                    "step_s": update_end - start_time,
                    "update_interval_s": update_end
                    - (previous_update_end or start_time),
                    "rollout_end_time": rollout_end,
                    "behavior_policy_version": behavior_version,
                    "training_policy_version": step - 1,
                    "policy_lag": policy_lag,
                    "sampling_s": sampling_s,
                    "forward_backward_s": forward_s,
                    "optimizer_s": optimizer_s,
                    "publication_s": publication_s,
                    "raw_rollouts": summary(groups),
                    "trained_response_tokens": trained_tokens,
                    "train_sample_logprob_diff_mean": float(np.abs(differences).mean()),
                    "train_sample_logprob_diff_p99": float(
                        np.quantile(np.abs(differences), 0.99)
                    ),
                    "train_sample_logprob_diff_max": float(np.abs(differences).max()),
                    "grad_norm": metrics["grad_norm:mean"],
                }
                previous_update_end = update_end
                report["client_steps"].append(row)
                print("GLM STEP", json.dumps(row), flush=True)
                await record(client, "step_complete", step)
                await asyncio.to_thread(volume.commit)
                del groups, old_logprobs, differences, output
                final = pending is None
                if not config.get("functional_validation") and (
                    step == 1 or step % config["checkpoint_every"] == 0 or final
                ):
                    await record(client, "checkpointing", step)
                    checkpoint = await submit(
                        OperationKind.SAVE_WEIGHTS,
                        SaveCheckpointPayload(destination=f"step-{step:04d}"),
                    )
                    checkpoints[client] = {
                        "client": client,
                        "step": step,
                        "cursor": cursor,
                        "checkpoint": checkpoint,
                    }
                    async with save_lock:
                        write_json(
                            root / f"client{client}-resume.json",
                            {"config": config, **checkpoints[client]},
                        )
                        await asyncio.to_thread(volume.commit)
                if not config.get("functional_validation") and (
                    step % config["eval_every"] == 0 or final
                ):
                    await record(client, "evaluation", step)
                    evaluation = await asyncio.to_thread(
                        sample_problems,
                        sampler,
                        [(client, prompt) for prompt in dataset["eval"]],
                        local_config,
                        step,
                        evaluation=True,
                    )
                    await asyncio.to_thread(
                        save_rollouts,
                        root / f"client{client}-eval-{step:04d}.jsonl.gz",
                        evaluation,
                    )
                    report["evaluations"].append(
                        {"client": client, "step": step, **summary(evaluation)}
                    )
                if final:
                    break
            await record(
                client, "completed" if step == config["steps"] else "paused", step
            )
        finally:
            for task in (pending, saving):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (pending, saving) if task is not None),
                return_exceptions=True,
            )
            await engine.unload_model(model)

    async with asyncio.TaskGroup() as tasks:
        for client in range(config["clients"]):
            tasks.create_task(client_loop(client))
    if config.get("functional_validation") and config["clients"] > 1:
        first = [r for r in report["client_steps"] if r["step"] == 1]
        second = [r for r in report["client_steps"] if r["step"] == 2]
        assert any(
            a["start_time"] < b["end_time"]
            for a in second
            for b in first
            if a["client"] != b["client"]
        ), "Clients did not overlap across steps"
        by_client = {(r["client"], r["step"]): r for r in report["client_steps"]}
        for client in range(config["clients"]):
            for step in range(2, config["steps"] + 1):
                previous = by_client[client, step - 1]
                current = by_client[client, step]
                assert current["start_time"] < previous["end_time"], (
                    "Client did not overlap its own rollout and update"
                )
                assert current["behavior_policy_version"] == step - 2
    write_json(root / "report.json", report)
    await asyncio.to_thread(volume.commit)
    return report


def train(settings, sampler, config):
    if config["resume"] or config.get("continue_run"):
        raise ValueError("The independent-client recipe requires a fresh run")
    torch.manual_seed(config["seed"])
    root = Path("/checkpoints") / config["run_id"]
    volume = modal.Volume.from_name(config["results_volume"])
    volume.reload()
    dataset = json.loads((root / "dataset.json").read_text())
    settings = {**settings, "hf_checkpoint": "/validation/model"}
    settings["cli_options"] = {
        **settings["cli_options"],
        "global_batch_size": config["groups"] * config["group_size"],
        "micro_batch_size": 1,
    }
    backend = MilesCommandBackend(
        MilesBackendConfig(**settings),
        checkpoint_dir=Path("/checkpoints"),
        capture_dir=Path("/tmp/captures"),
        base_model=MODEL,
    )
    executor = DistributedExecutor(backend)

    async def run():
        engine = Engine(
            executor, max_models=config["clients"], sampler_persistence_concurrency=1
        )
        try:
            validation = {
                **config,
                "run_id": config["run_id"] + "-validation",
                "steps": 3,
                "groups": 1,
                "group_size": 2,
                "max_tokens": 256,
                "stagger_s": 2,
                "concurrency": config["clients"] * 2,
                "functional_validation": True,
            }
            await run_clients(
                engine, sampler, validation, dataset, root / "validation", volume
            )
            print(
                "GLM VALIDATION PASSED: eight-slot memory, gates, routing replay, publication, per-client async overlap",
                flush=True,
            )
            write_json(
                root / "validation-passed.json",
                {
                    "time": time.time(),
                    "steps_per_client": 3,
                    "clients": config["clients"],
                    "async_rl": True,
                },
            )
            await asyncio.to_thread(volume.commit)
            if config.get("validation_only"):
                return {"run_id": config["run_id"], "validation_passed": True}
            # Validation models have been unloaded, so the real run starts with
            # fresh adapter weights and optimizer state on the warm base model.
            await run_clients(engine, sampler, config, dataset, root, volume)
            return {"run_id": config["run_id"], "report": str(root / "report.json")}
        finally:
            await engine.close()
            await executor.close()

    return asyncio.run(run())
