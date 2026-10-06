"""Packed multi-LoRA DAPO training, invoked by run_glm53_dapo.py."""

import json
import math
import time
from pathlib import Path

import modal
import numpy as np
import torch
from tinker import AdamParams, LoraConfig

from glm53_dapo_data import (
    MODEL,
    evaluate,
    group_advantages,
    publish,
    sample_problems,
    save_checkpoint,
    save_rollouts,
    summary,
    training_data,
    write_json,
    write_status,
)
from spindle.backends.contract import ForwardBatch, ForwardItem, ModelSpec
from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_lora import MilesCommandBackend


def train(settings, sampler, config):
    training_started = time.monotonic()
    torch.manual_seed(config["seed"])
    root = Path("/checkpoints") / config["run_id"]
    volume = modal.Volume.from_name(config["results_volume"])
    volume.reload()
    dataset = json.loads((root / "dataset.json").read_text())
    settings = dict(settings)
    settings["hf_checkpoint"] = "/validation/model"
    settings["cli_options"] = {
        **settings["cli_options"],
        "global_batch_size": config["clients"]
        * config["groups"]
        * config["group_size"],
        "micro_batch_size": 1,
    }
    start_step, cursors = 0, [0] * config["clients"]
    resume = None
    if config["resume"]:
        resume = json.loads((root / "resume.json").read_text())
        start_step, cursors = resume["step"], resume["cursors"]
    report = {"config": config, "steps": [], "evaluations": []}
    if not resume:
        report["evaluations"].append(json.loads((root / "baseline.json").read_text()))
    write_status(config, "starting_trainer")
    start = time.monotonic()
    backend = MilesCommandBackend(
        MilesBackendConfig(**settings),
        checkpoint_dir=Path("/checkpoints"),
        capture_dir=Path("/tmp/captures"),
        base_model=MODEL,
    )
    report["startup_s"] = time.monotonic() - start
    models = [
        f"{config['run_id']}-client{client}" for client in range(config["clients"])
    ]
    try:
        for client, model in enumerate(models):
            backend.accept_model(
                model,
                ModelSpec(
                    base_model=MODEL,
                    parameterization="lora",
                    lora_config=LoraConfig(rank=32, train_unembed=False),
                ),
            )
            if resume:
                backend.load_checkpoint(
                    model, resume["checkpoints"][client], restore_optimizer=True
                )
        if resume:
            publish(backend, sampler, config, start_step, dataset["eval"][0]["tokens"])
            report["evaluations"].append(evaluate(sampler, config, dataset, start_step))
        write_json(root / "report.json", report)
        volume.commit()
        for step in range(start_step + 1, config["steps"] + 1):
            step_start = time.monotonic()
            round_start_cursors = list(cursors)
            selected = [[] for _ in models]
            candidates = [[] for _ in models]
            wave_groups = max(1, config["concurrency"] // config["group_size"])
            while any(len(groups) < config["groups"] for groups in selected):
                if time.monotonic() - training_started > 72000:
                    completed_step = step - 1
                    if not (root / f"resume-{completed_step:04d}.json").exists():
                        save_checkpoint(
                            backend, config, completed_step, round_start_cursors
                        )
                    write_status(
                        config,
                        "paused",
                        step=completed_step,
                        resume=str(root / f"resume-{completed_step:04d}.json"),
                    )
                    return {
                        "run_id": config["run_id"],
                        "paused": True,
                        "step": completed_step,
                    }
                jobs, picked = [], [0] * len(models)
                while len(jobs) < wave_groups:
                    before = len(jobs)
                    for client in range(len(models)):
                        if (
                            len(jobs) == wave_groups
                            or len(selected[client]) + picked[client]
                            >= config["groups"]
                        ):
                            continue
                        if (
                            len(candidates[client]) + picked[client]
                            >= config["max_candidate_groups"]
                        ):
                            if picked[client] == 0:
                                raise RuntimeError(
                                    f"Client {client}: only {len(selected[client])} useful groups after {len(candidates[client])} candidates"
                                )
                            continue
                        cursor = cursors[client]
                        if cursor >= len(dataset["orders"][client]):
                            raise RuntimeError(
                                "Training dataset exhausted; refusing to silently repeat prompts"
                            )
                        index = dataset["orders"][client][cursor]
                        jobs.append((client, dataset["train"][index]))
                        cursors[client] += 1
                        picked[client] += 1
                    if len(jobs) == before:
                        break
                for group in sample_problems(sampler, jobs, config, step - 1):
                    client = group["client"]
                    candidates[client].append(group)
                    if np.any(group_advantages(group["samples"])):
                        selected[client].append(group)
                progress = []
                for client in range(len(models)):
                    save_rollouts(
                        root / f"client{client}-train-{step:04d}.jsonl.gz",
                        candidates[client],
                    )
                    progress.append(
                        {
                            "client": client,
                            "selected_groups": len(selected[client]),
                            **(
                                summary(candidates[client])
                                if candidates[client]
                                else {}
                            ),
                        }
                    )
                write_status(
                    config, "sampling", step=step, clients=progress, cursors=cursors
                )
            sampling_s = time.monotonic() - step_start
            batches = [training_data(groups) for groups in selected]
            write_status(
                config,
                "forward_backward",
                step=step,
                clients=len(models),
                trained_tokens=sum(tokens for _, _, tokens in batches),
            )
            start = time.monotonic()
            outputs = backend.forward_backward(
                ForwardBatch(
                    items=tuple(
                        ForwardItem(model, tuple(data))
                        for model, (data, _, _) in zip(models, batches, strict=True)
                    ),
                    loss_fn="ppo",
                    loss_fn_config={
                        "clip_low_threshold": config["clip_low"],
                        "clip_high_threshold": config["clip_high"],
                    },
                )
            )
            forward_s = time.monotonic() - start
            rows = []
            for client, (output, (_, replay, trained_tokens)) in enumerate(
                zip(outputs, batches, strict=True)
            ):
                differences = np.concatenate(
                    [
                        np.array(result["logprobs"].data[prefix:]) - np.array(old)
                        for result, (prefix, old) in zip(
                            output.loss_fn_outputs, replay, strict=True
                        )
                    ]
                )
                assert np.isfinite(differences).all()
                assert (
                    np.abs(differences).mean() < 0.5
                ), f"Client {client}: trainer/sampler logprobs disagree"
                rows.append(
                    {
                        "client": client,
                        "cursor": cursors[client],
                        "raw_rollouts": summary(candidates[client]),
                        "selected_rollouts": summary(selected[client]),
                        "trained_response_tokens": trained_tokens,
                        "train_sample_logprob_diff_mean": float(
                            np.abs(differences).mean()
                        ),
                        "train_sample_logprob_diff_p99": float(
                            np.quantile(np.abs(differences), 0.99)
                        ),
                        "train_sample_logprob_diff_max": float(
                            np.abs(differences).max()
                        ),
                        "initial_ratio_clip_fraction": float(
                            np.mean(
                                (differences < math.log(config["clip_low"]))
                                | (differences > math.log(config["clip_high"]))
                            )
                        ),
                    }
                )
            start = time.monotonic()
            optimizers = backend.optim_step(
                tuple(models),
                AdamParams(learning_rate=config["learning_rate"], grad_clip_norm=1.0),
            )
            optimizer_s = time.monotonic() - start
            for row, optimizer in zip(rows, optimizers, strict=True):
                assert optimizer.metrics["update_successful:mean"] == 1
                norm = optimizer.metrics["grad_norm:mean"]
                assert math.isfinite(norm) and norm > 0
                row["grad_norm"] = norm
            round_metrics = {
                "step": step,
                "clients": rows,
                "sampling_s": sampling_s,
                "forward_backward_s": forward_s,
                "optimizer_s": optimizer_s,
            }
            if (
                step == 1
                or step % config["checkpoint_every"] == 0
                or step == config["steps"]
            ):
                write_status(config, "checkpointing", step=step)
                round_metrics["checkpoint_s"] = save_checkpoint(
                    backend, config, step, cursors
                )
            write_status(config, "publishing", step=step)
            round_metrics["publication"] = publish(
                backend, sampler, config, step, dataset["eval"][0]["tokens"]
            )
            round_metrics["step_s"] = time.monotonic() - step_start
            report["steps"].append(round_metrics)
            write_json(root / "report.json", report)
            volume.commit()
            print("DAPO STEP", json.dumps(round_metrics), flush=True)
            if step % config["eval_every"] == 0 or step == config["steps"]:
                write_status(config, "evaluation", step=step)
                report["evaluations"].append(evaluate(sampler, config, dataset, step))
                write_json(root / "report.json", report)
                volume.commit()
            if step < config["steps"] and time.monotonic() - training_started > 72000:
                # Leave time for durable checkpoints before Modal's 24-hour call limit.
                if "checkpoint_s" not in round_metrics:
                    save_checkpoint(backend, config, step, cursors)
                write_status(
                    config, "paused", step=step, resume=str(root / "resume-latest.json")
                )
                return {"run_id": config["run_id"], "paused": True, "step": step}
        write_status(config, "completed", step=config["steps"])
        return {"run_id": config["run_id"], "report": str(root / "report.json")}
    finally:
        backend.close()
