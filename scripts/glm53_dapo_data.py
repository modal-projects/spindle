"""DAPO Math rollout, reward, training, evaluation, and recovery for GLM LoRA.

Launched by run_glm53_dapo.py. Uses Spindle's Miles command backend directly;
SDK ingress and EngineServer scheduling are outside this experiment.
"""

import gzip
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import numpy as np
from huggingface_hub import hf_hub_download
from miles.rollout.rm_hub.math_dapo_utils import compute_score
from tinker import Datum, ModelInput, TensorData
from transformers import AutoTokenizer


MODEL = "zai-org/GLM-5.3-Flash"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_status(config, phase, **details):
    row = {"phase": phase, "time": time.time(), "run_id": config["run_id"], **details}
    write_json(Path("/checkpoints") / config["run_id"] / "status.json", row)
    modal.Volume.from_name(config["results_volume"]).commit()
    print("DAPO STATUS", json.dumps(row), flush=True)


def prepare(config):
    root = Path("/checkpoints") / config["run_id"]
    root.mkdir(exist_ok=True)
    previous_path = root / "config.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for key in (
            "clients",
            "groups",
            "group_size",
            "max_tokens",
            "learning_rate",
            "seed",
            "dataset",
            "dataset_revision",
            "eval_prompts",
            "clip_low",
            "clip_high",
        ):
            if previous[key] != config[key]:
                raise ValueError(f"Existing run differs for {key}")
        if (root / "dataset.json").exists():
            write_json(root / "config.json", config)
            modal.Volume.from_name(config["results_volume"]).commit()
            return
    elif config.get("continue_run"):
        raise ValueError("The requested prior run does not exist")
    write_json(root / "config.json", config)
    write_status(config, "preparing_dataset")
    path = hf_hub_download(
        config["dataset"],
        "dapo-math-17k.jsonl",
        repo_type="dataset",
        revision=config["dataset_revision"],
    )
    tokenizer = AutoTokenizer.from_pretrained("/validation/model-bf16")
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    order = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(order)
    prepared, skipped = [], 0
    for index in order:
        row = rows[index]
        tokens = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        assert (
            tokenizer.decode(tokens[-8:]).rstrip().endswith("<think>")
        ), "Expected GLM native reasoning prefix"
        if len(tokens) + config["max_tokens"] > config["context_length"]:
            skipped += 1
            continue
        prepared.append(
            {
                "id": index,
                "messages": row["prompt"],
                "answer": str(row["label"]),
                "tokens": tokens,
            }
        )
    # Split by prompt content too, so duplicate dataset rows cannot cross the split.
    evaluation = prepared[: config["eval_prompts"]]
    eval_text = {json.dumps(row["messages"], sort_keys=True) for row in evaluation}
    training = [
        row
        for row in prepared[config["eval_prompts"] :]
        if json.dumps(row["messages"], sort_keys=True) not in eval_text
    ]
    assert (
        len(evaluation) == config["eval_prompts"] and len(training) >= config["groups"]
    )
    orders = []
    for client in range(config["clients"]):
        order = list(range(len(training)))
        random.Random(config["seed"] + 1009 * client).shuffle(order)
        orders.append(order)
    write_json(
        root / "dataset.json", {"eval": evaluation, "train": training, "orders": orders}
    )
    if config["resume"]:
        state = json.loads(Path(config["resume"]).read_text())
        previous = state["config"]
        for key in (
            "clients",
            "dataset",
            "dataset_revision",
            "seed",
            "groups",
            "group_size",
            "max_tokens",
            "eval_prompts",
            "enable_thinking",
            "learning_rate",
            "clip_low",
            "clip_high",
        ):
            if previous[key] != config[key]:
                raise ValueError(f"Resume config differs for {key}")
        assert state["step"] < config["steps"]
        write_json(root / "resume.json", state)
    write_status(
        config,
        "waiting_for_sampler",
        train_prompts=len(training),
        eval_prompts=len(evaluation),
        skipped_long_prompts=skipped,
    )


def grade_sample(sample, problem):
    tokens = sample["output_ids"]
    metadata = sample["meta_info"]
    reason = metadata["finish_reason"]["type"]
    if reason not in ("stop", "length"):
        raise RuntimeError(f"Unexpected sample finish: {metadata['finish_reason']}")
    probabilities = metadata["output_token_logprobs"]
    assert tokens and len(tokens) == len(probabilities)
    assert [
        row[1] for row in probabilities
    ] == tokens, "Returned logprobs do not match generated tokens"
    logprobs = [row[0] for row in probabilities]
    assert all(math.isfinite(value) for value in logprobs)
    score = compute_score(sample["text"], problem["answer"])
    return {
        "tokens": tokens,
        "logprobs": logprobs,
        "text": sample["text"],
        "reward": float(score["score"]),
        "correct": bool(score["acc"]),
        "prediction": score["pred"],
        "truncated": reason == "length",
        "served_version": metadata.get("weight_version_start", 0),
    }


def group_advantages(samples):
    # Incomplete answers do not receive policy-gradient updates. They still count
    # in the raw accuracy and truncation metrics, and their compute is accounted for.
    active = np.array([not row["truncated"] for row in samples])
    rewards = np.array([row["reward"] for row in samples], dtype=np.float64)
    advantages = np.zeros(len(samples))
    if active.sum() >= 2 and np.ptp(rewards[active]) > 0:
        centered = rewards[active] - rewards[active].mean()
        advantages[active] = centered / (rewards[active].std(ddof=1) + 1e-6)
    return advantages


def training_data(groups):
    total_tokens = sum(
        len(s["tokens"]) for g in groups for s in g["samples"] if not s["truncated"]
    )
    assert total_tokens > 0
    data, replay = [], []
    for group in groups:
        prompt = group["problem"]["tokens"]
        for sample, advantage in zip(
            group["samples"], group_advantages(group["samples"]), strict=True
        ):
            tokens = prompt + sample["tokens"]
            prefix = len(prompt) - 1
            length = len(tokens) - 1
            data.append(
                Datum(
                    ModelInput.from_ints(tokens[:-1]),
                    {
                        "target_tokens": TensorData(
                            data=tokens[1:], dtype="int64", shape=[length]
                        ),
                        "logprobs": TensorData(
                            data=[0.0] * prefix + sample["logprobs"],
                            dtype="float32",
                            shape=[length],
                        ),
                        # Miles accumulates sums; normalize across the whole update.
                        "advantages": TensorData(
                            data=[0.0] * prefix
                            + [float(advantage) / total_tokens] * len(sample["tokens"]),
                            dtype="float32",
                            shape=[length],
                        ),
                    },
                )
            )
            replay.append((prefix, sample["logprobs"]))
    return data, replay, total_tokens


def sample_problems(sampler, jobs, config, version, *, evaluation=False, on_group=None):
    """Mix clients in one request stream; every request pins its own adapter."""
    requests = []
    count = 1 if evaluation else config["group_size"]
    for client, problem in jobs:
        for member in range(count):
            payload = {
                "input_ids": problem["tokens"],
                "return_logprob": True,
                "sampling_params": {
                    "max_new_tokens": config["max_tokens"],
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    # Hold evaluation seeds fixed across adapters and versions.
                    "sampling_seed": config["seed"]
                    + problem["id"] * config["group_size"]
                    + member
                    + (0 if evaluation else 1000003 * client),
                },
            }
            if version:
                payload.update(
                    weight_run_id=f"{config['run_id']}-client{client}",
                    weight_version={"exact_version": version},
                )
            requests.append(payload)
    groups = [
        {"client": client, "problem": problem, "samples": [None] * count}
        for client, problem in jobs
    ]
    remaining = [count] * len(jobs)
    pool = ThreadPoolExecutor(max_workers=config["concurrency"])
    futures = {
        pool.submit(sampler.request.remote, "generate", payload): divmod(index, count)
        for index, payload in enumerate(requests)
    }
    try:
        for future in as_completed(futures):
            index, member = futures[future]
            group = groups[index]
            sample = grade_sample(future.result(), group["problem"])
            assert (
                sample["served_version"] == version
            ), "Sampler served the wrong policy version"
            group["samples"][member] = sample
            remaining[index] -= 1
            if remaining[index] == 0 and on_group is not None:
                on_group(group)
    finally:
        # On failure, let the controller terminate serving instead of waiting for
        # every queued generation to finish before it can clean up.
        pool.shutdown(wait=False, cancel_futures=True)
    return groups


def collect_training_groups(sampler, dataset, config, step, cursors, root):
    """Collect a fixed batch, retaining constant-reward groups without replacement."""
    groups = [[] for _ in range(config["clients"])]
    positions = {}
    jobs = []
    for client in range(config["clients"]):
        end = cursors[client] + config["groups"]
        if end > len(dataset["orders"][client]):
            raise RuntimeError(
                "Training dataset exhausted; refusing to silently repeat prompts"
            )
        for position in range(cursors[client], end):
            index = dataset["orders"][client][position]
            problem = dataset["train"][index]
            positions[client, problem["id"]] = position
        path = root / f"client{client}-train-{step:04d}.jsonl.gz"
        if config.get("continue_run") and step == 1 and path.exists():
            with gzip.open(path, "rt") as stream:
                groups[client] = [json.loads(line) for line in stream]
            seen = set()
            for group in groups[client]:
                key = (client, group["problem"]["id"])
                assert group["client"] == client and key in positions
                position = positions[key]
                assert position not in seen, "Duplicate saved training group"
                seen.add(position)
                expected = dataset["train"][dataset["orders"][client][position]]
                assert group["problem"] == expected, "Saved prompt differs from dataset"
                assert len(group["samples"]) == config["group_size"]
                assert all(sample["served_version"] == 0 for sample in group["samples"])
                group["position"] = position
        else:
            seen = set()
        for position in range(cursors[client], end):
            if position not in seen:
                index = dataset["orders"][client][position]
                jobs.append((client, dataset["train"][index]))
    # Interleave clients so each has requests admitted throughout collection.
    jobs.sort(key=lambda job: (positions[job[0], job[1]["id"]], job[0]))
    last_saved = 0.0

    def persist():
        nonlocal last_saved
        progress = []
        for client, completed in enumerate(groups):
            completed.sort(key=lambda group: group["position"])
            save_rollouts(root / f"client{client}-train-{step:04d}.jsonl.gz", completed)
            progress.append(
                {
                    "client": client,
                    "completed_groups": len(completed),
                    "target_groups": config["groups"],
                    "informative_groups": sum(
                        bool(np.any(group_advantages(g["samples"]))) for g in completed
                    ),
                    **(summary(completed) if completed else {}),
                }
            )
        write_status(config, "sampling", step=step, clients=progress)
        last_saved = time.monotonic()

    def completed(group):
        group["position"] = positions[group["client"], group["problem"]["id"]]
        groups[group["client"]].append(group)
        if time.monotonic() - last_saved >= 60:
            persist()

    persist()
    try:
        sample_problems(sampler, jobs, config, step - 1, on_group=completed)
    finally:
        persist()
    assert all(len(batch) == config["groups"] for batch in groups)
    for client in range(config["clients"]):
        cursors[client] += config["groups"]
    return groups


def summary(groups):
    samples = [s for group in groups for s in group["samples"]]
    lengths = np.array([len(s["tokens"]) for s in samples])
    return {
        "groups": len(groups),
        "responses": len(samples),
        "accuracy": float(
            np.mean([s["correct"] and not s["truncated"] for s in samples])
        ),
        "reward_mean": float(np.mean([s["reward"] for s in samples])),
        "truncated_fraction": float(np.mean([s["truncated"] for s in samples])),
        "generation_length_mean": float(lengths.mean()),
        "generation_length_p95": float(np.quantile(lengths, 0.95)),
        "generation_length_max": int(lengths.max()),
        "generated_tokens": int(lengths.sum()),
    }


def save_rollouts(path, groups):
    temporary = Path(str(path) + ".tmp")
    with gzip.open(temporary, "wt") as f:
        for group in groups:
            f.write(json.dumps(group) + "\n")
    temporary.replace(path)


def evaluate(sampler, config, dataset, step):
    root = Path("/checkpoints") / config["run_id"]
    start = time.monotonic()
    # All adapters start with zero B matrices. Evaluate the shared base once.
    clients = range(config["clients"]) if step else range(1)
    jobs = [(client, problem) for problem in dataset["eval"] for client in clients]
    groups = sample_problems(sampler, jobs, config, step, evaluation=True)
    save_rollouts(root / f"eval-{step:04d}.jsonl.gz", groups)
    row = {"step": step, "seconds": time.monotonic() - start, "clients": []}
    for client in range(config["clients"]):
        selected = [g for g in groups if g["client"] == (client if step else 0)]
        row["clients"].append({"client": client, **summary(selected)})
    print("DAPO EVAL", json.dumps(row), flush=True)
    return row


def baseline(sampler, config):
    root = Path("/checkpoints") / config["run_id"]
    dataset = json.loads((root / "dataset.json").read_text())
    write_status(config, "baseline_evaluation", step=0)
    row = evaluate(sampler, config, dataset, 0)
    write_json(root / "baseline.json", row)
    modal.Volume.from_name(config["results_volume"]).commit()
    if row["clients"][0]["truncated_fraction"] > 0.8:
        raise RuntimeError(
            "More than 80% of baseline responses hit the generation cap; inspect outputs before allocating trainers"
        )


def publish(backend, sampler, config, step, probe_tokens):
    timings = []
    for client in range(config["clients"]):
        model = f"{config['run_id']}-client{client}"
        start = time.monotonic()
        capture = f"{model}-weights-{step:04d}"
        backend.capture_sampler_snapshot(model, capture, step)
        row = {"client": client, "capture_s": time.monotonic() - start}
        start = time.monotonic()
        backend.publish_sampler_snapshot(capture)
        row["publication_s"] = time.monotonic() - start
        timings.append(row)
    # Register all adapters before normal mixed serving. The sidecar coordinates
    # registration and volume refresh; requests specify both run ID and version.
    for row in timings:
        start = time.monotonic()
        probe = sampler.request.remote(
            "generate",
            {
                "input_ids": probe_tokens,
                "weight_run_id": f"{config['run_id']}-client{row['client']}",
                "weight_version": {"exact_version": step},
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
        )
        assert probe["meta_info"]["weight_version_start"] == step
        row["load_and_probe_s"] = time.monotonic() - start
    return timings


def save_checkpoint(backend, config, step, cursors):
    start = time.monotonic()
    checkpoints = []
    for client in range(config["clients"]):
        model = f"{config['run_id']}-client{client}"
        snapshot = f"{model}-state-{step:04d}"
        destination = f"{config['run_id']}/checkpoints/step-{step:04d}"
        backend.capture_checkpoint(
            model, snapshot, destination=destination, include_optimizer=True
        )
        checkpoints.append(backend.persist_checkpoint(snapshot, destination))
    root = Path("/checkpoints") / config["run_id"]
    state = {
        "config": config,
        "step": step,
        "cursors": cursors,
        "checkpoints": checkpoints,
    }
    write_json(root / f"resume-{step:04d}.json", state)
    write_json(root / "resume-latest.json", state)
    modal.Volume.from_name(config["results_volume"]).commit()
    return time.monotonic() - start
