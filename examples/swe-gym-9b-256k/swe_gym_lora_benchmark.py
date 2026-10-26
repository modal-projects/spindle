"""One versus eight independent asynchronous SWE-Gym DAPO clients on one trainer.

All generated tokens/logprobs are retained verbatim. Only generated tokens carry
loss. Tool observations and prompts never receive policy-gradient credit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gzip
import json
import math
import os
import queue
import random
import re
import shlex
import threading
import time
from pathlib import Path

import orjson
import numpy as np
import tinker
from tinker import types
from transformers import AutoTokenizer

from swe_gym_env import Environments, InfrastructureError, command

MODEL = "Qwen/Qwen3.5-9B"
URL = os.environ.get("SPINDLE_BASE_URL")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in /testbed to inspect, edit, and test the repository.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
SYSTEM = """You are a software engineer fixing a repository issue. Use the bash tool
to inspect and edit /testbed, and run relevant tests. The testbed conda environment
is activated for every command. All shell commands are isolated and have no network
access. Make a minimal correct source fix; do not weaken or delete tests. When the
fix is complete, respond with a short final explanation without a tool call."""


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def engine_lost(exc):
    return "model_lost" in str(exc) or "engine owning model" in str(exc)


def request_restart(log, job, exc):
    # Multiple clients can observe the same preemption simultaneously.
    with log.lock:
        path = log.root / "RESTART_REQUIRED.json"
        if not path.exists():
            atomic_json(
                path,
                {
                    "time": time.time(),
                    "job": job,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


class Log:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.sample_windows = {}

    def event(self, kind, **fields):
        row = {"time": time.time(), "event": kind, **fields}
        with self.lock:
            with (self.root / "events.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            if kind in {
                "step_done",
                "step_interval",
                "optimizer_step",
                "phase_start",
                "phase_end",
                "checkpoint_restored",
            }:
                with (self.root / "training_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            if kind == "trainers_ready":
                atomic_json(self.root / "trainers.json", row)
            if kind == "phase_start":
                atomic_json(self.root / "phase.json", row)
            if kind == "sample":
                key = (fields["phase"], fields["job"])
                window = self.sample_windows.setdefault(
                    key,
                    {"start": row["time"], "tokens": 0, "requests": 0, "sample_s": 0.0},
                )
                window["tokens"] += fields["tokens"]
                window["requests"] += 1
                window["sample_s"] += fields["sample_s"]
                duration = row["time"] - window["start"]
                if duration >= 60:
                    metric = dict(
                        time=row["time"],
                        event="sampling_window",
                        phase=key[0],
                        job=key[1],
                        duration_s=duration,
                        output_tps=window["tokens"] / duration,
                        mean_sample_s=window["sample_s"] / window["requests"],
                        **window,
                    )
                    with (self.root / "training_metrics.jsonl").open("a") as f:
                        f.write(json.dumps(metric) + "\n")
                    del self.sample_windows[key]
            if kind not in {
                "sample",
                "tool",
                "episode",
                "group",
                "rollout_start",
                "rollout_ready",
                "grading_start",
            }:
                atomic_json(self.root / "status.json", row)
        if kind not in {
            "sample",
            "tool",
            "rollout_start",
            "rollout_ready",
            "grading_start",
        }:
            print(json.dumps(row, allow_nan=False), flush=True)
        return row


def parse_commands(text):
    # Qwen3.5 uses XML function/parameter tags, not the Qwen3 JSON tool format.
    text = text.split("</think>")[-1]
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    result = []
    for call in calls:
        match = re.fullmatch(
            r"<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>",
            call,
            re.S,
        )
        if not match:
            raise ValueError("Expected bash(command) in the documented XML tool format")
        result.append(match.group(1))
    if "<tool_call>" in text and not calls:
        raise ValueError("Incomplete tool call")
    return result


def generation_budget(cfg, prompt_tokens):
    # Reference-logprob requests also need backend-reserved tokens. Keep the
    # complete sampled trajectory below the configured reference context limit.
    return min(cfg["turn_tokens"], cfg["context_tokens"] - prompt_tokens - 16)


def generate_rollout(inst, sampler, tokenizer, environments, cfg, log, identity):
    start = time.time()
    log.event("rollout_start", **identity, instance_id=inst["instance_id"])
    path = (
        log.root
        / "episodes"
        / identity["phase"]
        / str(identity["job"])
        / f"g{identity['group']:06d}-a{identity['attempt']}"
    )
    path.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": inst["problem_statement"]},
    ]
    tokens = list(
        tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )
    behavior = [0.0] * len(tokens)
    mask = [0] * len(tokens)
    turns = []
    termination = "turn_limit"
    with environments.sandbox(inst) as sb:
        environments.prepare_agent(sb)
        for turn in range(cfg["max_turns"]):
            budget = generation_budget(cfg, len(tokens))
            if budget < 1:
                termination = "context_limit"
                break
            sample_start = time.time()
            sample = (
                sampler.sample(
                    types.ModelInput.from_ints(tokens),
                    1,
                    types.SamplingParams(
                        max_tokens=budget, temperature=cfg["temperature"], top_p=1.0
                    ),
                )
                .result(timeout=3600)
                .sequences[0]
            )
            generated, lp = list(sample.tokens), list(sample.logprobs or [])
            if (
                not generated
                or len(lp) != len(generated)
                or not all(math.isfinite(x) for x in lp)
            ):
                raise InfrastructureError(
                    "Missing/nonfinite sampling log probabilities"
                )
            sample_s = time.time() - sample_start
            text = tokenizer.decode(generated, skip_special_tokens=False)
            row = {
                "turn": turn,
                "prompt_tokens": len(tokens),
                "tokens": len(generated),
                "sample_s": sample_s,
                "text": text,
                "stop_reason": str(sample.stop_reason),
            }
            turns.append(row)
            log.event(
                "sample", **identity, **{k: v for k, v in row.items() if k != "text"}
            )
            tokens.extend(generated)
            behavior.extend(lp)
            mask.extend([1] * len(generated))
            if len(generated) >= budget:
                termination = "generation_limit"
                break
            try:
                cmds = parse_commands(text)
                feedback = []
                for cmd in cmds:
                    ts = time.time()
                    rc, output = command(
                        sb,
                        "source /opt/miniconda3/bin/activate testbed && cd /testbed && "
                        + "bash -c "
                        + shlex.quote(cmd),
                        120,
                    )
                    feedback.append(f"exit_code={rc}\n{output[-20000:]}")
                    log.event(
                        "tool",
                        **identity,
                        turn=turn,
                        elapsed_s=time.time() - ts,
                        exit_code=rc,
                    )
                if not cmds:
                    termination = "final"
                    break
            except ValueError as exc:
                feedback = [str(exc)]
            # Preserve sampled token IDs exactly; append template-compatible tool
            # observations. Re-rendering the entire history can rewrite reasoning.
            suffix = (
                ""
                if tokens[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>")
                else "<|im_end|>"
            )
            suffix += "\n<|im_start|>user"
            for output in feedback:
                suffix += "\n<tool_response>\n" + output.strip() + "\n</tool_response>"
            suffix += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            obs = tokenizer.encode(suffix, add_special_tokens=False)
            tokens.extend(obs)
            behavior.extend([0.0] * len(obs))
            mask.extend([0] * len(obs))
        # Persist the trajectory before patch extraction or grading can fail.
        write_episode(
            path / "ungraded.json.gz",
            {
                **identity,
                "instance_id": inst["instance_id"],
                "tokens": tokens,
                "behavior_logprobs": behavior,
                "mask": mask,
                "turns": turns,
                "termination": termination,
            },
        )
        patch = environments.patch(sb)
    # Drop trailing observations: every datum ends with a generated target token.
    while mask and not mask[-1]:
        tokens.pop()
        behavior.pop()
        mask.pop()
    episode = {
        **identity,
        "instance_id": inst["instance_id"],
        "tokens": tokens,
        "behavior_logprobs": behavior,
        "mask": mask,
        "turns": turns,
        "termination": termination,
        "generated_tokens": sum(mask),
        "modal_image_id": environments.image(inst).object_id,
    }
    # Handoff only paths and small metadata; queued 128K trajectories stay on disk.
    (path / "prediction.patch").write_text(patch)
    write_episode(path / "ungraded.json.gz", episode)
    ready = time.time()
    log.event(
        "rollout_ready",
        **identity,
        instance_id=inst["instance_id"],
        generation_s=ready - start,
        generated_tokens=sum(mask),
    )
    return {
        "path": str(path),
        "identity": identity,
        "started_at": start,
        "ready_at": ready,
        "generation_s": ready - start,
    }


def grade_rollout(prepared, inst, environments, log):
    path, identity = Path(prepared["path"]), prepared["identity"]
    grade_start = time.time()
    queue_s = grade_start - prepared["ready_at"]
    log.event(
        "grading_start",
        **identity,
        instance_id=inst["instance_id"],
        grading_queue_s=queue_s,
    )
    verdict = environments.grade(inst, (path / "prediction.patch").read_text(), path)
    episode = load_episode(path / "ungraded.json.gz")
    episode.update(
        reward=verdict["reward"],
        elapsed_s=time.time() - prepared["started_at"],
        generation_s=prepared["generation_s"],
        grading_queue_s=queue_s,
        grading_s=time.time() - grade_start,
    )
    trace = path / "trajectory.json.gz"
    write_episode(trace, episode)
    log.event(
        "episode",
        **identity,
        instance_id=inst["instance_id"],
        reward=episode["reward"],
        generated_tokens=episode["generated_tokens"],
        context_tokens=len(episode["tokens"]),
        termination=episode["termination"],
        generation_s=episode["generation_s"],
        grading_queue_s=queue_s,
        elapsed_s=episode["elapsed_s"],
        grading_s=episode["grading_s"],
        trace=str(trace),
    )
    return {
        "path": str(trace),
        "reward": episode["reward"],
        "generated_tokens": episode["generated_tokens"],
        "termination": episode["termination"],
    }


def rollout(inst, sampler, tokenizer, environments, cfg, log, identity):
    """Synchronous convenience path for the single-episode smoke test."""
    prepared = generate_rollout(
        inst, sampler, tokenizer, environments, cfg, log, identity
    )
    return grade_rollout(prepared, inst, environments, log)


class GradingPool:
    """Bound running + queued grading jobs; only coordinator threads may wait."""

    def __init__(self, workers, queue_size):
        self.slots = threading.BoundedSemaphore(workers + queue_size)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="swe-grading")

    def submit(self, *args):
        self.slots.acquire()
        try:
            future = self.pool.submit(*args)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True)


def write_episode(path, value):
    with gzip.open(path, "wb", compresslevel=1) as stream:
        stream.write(orjson.dumps(value))


def load_episode(path):
    with gzip.open(path, "rb") as f:
        return orjson.loads(f.read())


def publish_policy(trainer, service, name):
    # Lilo's unnamed ephemeral sampling clients follow "latest". A named artifact
    # is required to hold the policy fixed throughout a multi-turn episode.
    receipt = trainer.save_weights_for_sampler(name=name).result(timeout=3600)
    return service.create_sampling_client(model_path=receipt.path)


def advantages(rewards):
    a = np.asarray(rewards, dtype=np.float64)
    return (a - a.mean()) / (a.std(ddof=1) + 1e-6)


def datum(episode, advantage, token_denominator, reference_logprobs, kl_coef):
    tokens, mask, old = episode["tokens"], episode["mask"], episode["behavior_logprobs"]
    if len(reference_logprobs) != len(tokens):
        raise InfrastructureError("Reference logprobs length mismatch")
    adv = []
    for i in range(1, len(tokens)):
        if mask[i]:
            ref = reference_logprobs[i]
            if ref is None or not math.isfinite(ref):
                raise InfrastructureError(
                    "Reference logprob missing on generated token"
                )
            # Frozen-reference KL reward shaping, evaluated at the behavior policy.
            # This is not an extra differentiable current-policy KL loss.
            adv.append(
                (float(advantage) - kl_coef * (old[i] - ref)) / token_denominator
            )
        else:
            adv.append(0.0)

    def tensor(data, dtype):
        return types.TensorData(data=data, dtype=dtype, shape=[len(data)])

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tensor(tokens[1:], "int64"),
            "logprobs": tensor(old[1:], "float32"),
            "advantages": tensor(adv, "float32"),
        },
    )


def logprob_difference(data, masks, outputs):
    """Token-weighted train-minus-behavior difference, before this optimizer step."""
    differences = []
    for item, mask, output in zip(data, masks, outputs, strict=True):
        current = np.asarray(output["logprobs"].data, dtype=np.float64)
        behavior = np.asarray(item.loss_fn_inputs["logprobs"].data, dtype=np.float64)
        generated = np.asarray(mask, dtype=bool)
        if current.shape != behavior.shape or current.shape != generated.shape:
            raise ValueError("Training/rollout logprob token alignment differs")
        differences.append((current - behavior)[generated])
    diff = np.concatenate(differences)
    if not len(diff) or not np.isfinite(diff).all():
        raise ValueError("Missing or nonfinite generated-token logprob differences")
    return {
        "tokens": int(diff.size),
        "mean": float(diff.mean()),
        "mean_absolute": float(np.abs(diff).mean()),
        "rms": float(np.sqrt(np.square(diff).mean())),
        "outside_clip_fraction": float(
            ((diff < math.log(0.8)) | (diff > math.log(1.28))).mean()
        ),
    }


class Job:
    def __init__(
        self, job, trainer, service, tokenizer, environments, tasks, cfg, log, reference
    ):
        self.job, self.trainer, self.service = job, trainer, service
        self.tokenizer, self.env, self.tasks = tokenizer, environments, tasks
        self.cfg, self.log, self.reference = cfg, log, reference
        self.lock = threading.Lock()
        self.version = 0
        self.sampler = None

    def run_group(self, inst, phase, group, sampler, version, pool, graders):
        identities = [
            {
                "phase": phase,
                "job": self.job,
                "group": group,
                "attempt": a,
                "policy_version": version,
            }
            for a in range(self.cfg["group_size"])
        ]
        futures = {
            pool.submit(
                generate_rollout,
                inst,
                sampler,
                self.tokenizer,
                self.env,
                self.cfg,
                self.log,
                ident,
            )
            for ident in identities
        }
        results, errors, graded = [], [], []
        # Submit each finished rollout immediately; one slow attempt must not
        # hold its siblings out of the grading pool. Any backpressure blocks a
        # group coordinator, never a generation worker.
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    graded.append(
                        graders.submit(
                            grade_rollout, f.result(), inst, self.env, self.log
                        )
                    )
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        for f in graded:
            try:
                results.append(f.result())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            self.log.event(
                "group_error", phase=phase, job=self.job, group=group, errors=errors
            )
            raise InfrastructureError(
                "Incomplete rollout group; saved successes are retained on disk"
            )
        rewards = [e["reward"] for e in results]
        mixed = 0 < sum(rewards) < len(rewards)
        self.log.event(
            "group",
            phase=phase,
            job=self.job,
            group=group,
            policy_version=version,
            instance_id=inst["instance_id"],
            rewards=rewards,
            mixed=mixed,
        )
        return {
            "episodes": results,
            "policy_version": version,
            "group": group,
            "mixed": mixed,
        }

    def producer(self, phase, output, stop):
        # Admit more groups than fit in the episode pool so workers freed by
        # fast attempts can continue while stragglers finish tools or grading.
        # Bound both admission and completed results to limit policy staleness.
        rng = random.Random(self.cfg["seed"] + self.job)
        order = list(self.tasks)
        rng.shuffle(order)
        group = getattr(self, "start_group", 0)
        limit = self.cfg["inflight_groups"]
        with (
            ThreadPoolExecutor(
                self.cfg["rollout_workers"], thread_name_prefix="swe-rollout"
            ) as episodes,
            GradingPool(
                self.cfg["grading_workers"], self.cfg["grading_queue_size"]
            ) as graders,
            ThreadPoolExecutor(limit) as groups,
        ):
            pending = set()
            while not stop.is_set() or pending:
                while not stop.is_set() and len(pending) < limit and not output.full():
                    with self.lock:
                        sampler, version = self.sampler, self.version
                    inst = order[group % len(order)]
                    pending.add(
                        groups.submit(
                            self.run_group,
                            inst,
                            phase,
                            group,
                            sampler,
                            version,
                            episodes,
                            graders,
                        )
                    )
                    group += 1
                if not pending:
                    stop.wait(0.2)
                    continue
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = exc
                    while not stop.is_set():
                        try:
                            output.put(result, timeout=0.5)
                            break
                        except queue.Full:
                            pass

    def run(self, phase, deadline, start_step=0, start_group=0):
        if start_step >= self.cfg["steps"]:
            return
        self.version = start_step
        self.start_group = start_group
        recovery_path = self.log.root / f"recovered-groups-job{self.job}.json"
        recovery = (
            json.loads(recovery_path.read_text())
            if start_step == 0 and recovery_path.exists()
            else {}
        )
        recovered = recovery.get("groups", [])
        if self.sampler is None:
            self.sampler = publish_policy(
                self.trainer, self.service, f"{phase}-job{self.job}-initial"
            )
        output, stop = (
            queue.Queue(self.cfg["completed_group_buffer"]),
            threading.Event(),
        )
        producer = threading.Thread(
            target=self.producer, args=(phase, output, stop), daemon=True
        )
        producer.start()
        previous_start = None
        reference_pool = ThreadPoolExecutor(16, thread_name_prefix="swe-reference")
        try:
            for step in range(start_step, self.cfg["steps"]):
                started = (
                    recovery.get("original_started_at", time.time())
                    if step == 0
                    else time.time()
                )
                if previous_start is not None:
                    self.log.event(
                        "step_interval",
                        phase=phase,
                        job=self.job,
                        step=step - 1,
                        start_to_start_s=started - previous_start,
                    )
                previous_start = started
                self.log.event(
                    "step_start",
                    phase=phase,
                    job=self.job,
                    step=step,
                    batch_started_at=started,
                    recovered_groups=len(recovered),
                )
                chosen, errors, incoming = [], 0, []
                reference_futures = {}
                while len(chosen) < self.cfg["groups_per_batch"]:
                    if time.time() >= deadline:
                        raise TimeoutError("Benchmark wall time limit reached")
                    try:
                        result = (
                            recovered.pop(0) if recovered else output.get(timeout=10)
                        )
                    except queue.Empty:
                        continue
                    if isinstance(result, Exception):
                        errors += 1
                        if errors >= 5:
                            raise InfrastructureError(
                                "Five infrastructure-failed groups in one step"
                            ) from result
                        continue
                    incoming.extend(e["reward"] for e in result["episodes"])
                    if self.version - result["policy_version"] > 1:
                        self.log.event(
                            "stale_group",
                            phase=phase,
                            job=self.job,
                            group=result["group"],
                        )
                    elif result["mixed"]:
                        chosen.append(result)

                        # Fetch immutable base-policy logprobs while later groups
                        # finish. Persist them so the queue holds no large tensors.
                        def fetch_reference(saved):
                            episode = load_episode(saved["path"])
                            started_ref = time.time()
                            refs = self.reference.compute_logprobs(
                                types.ModelInput.from_ints(episode["tokens"])
                            ).result(timeout=3600)
                            dest = Path(saved["path"]).with_name("reference.json.gz")
                            with gzip.open(dest, "wb", compresslevel=1) as stream:
                                stream.write(orjson.dumps(refs))
                            self.log.event(
                                "reference_logprobs",
                                phase=phase,
                                job=self.job,
                                step=step,
                                seconds=time.time() - started_ref,
                                tokens=len(refs),
                            )
                            return str(dest)

                        for saved in result["episodes"]:
                            reference_futures[saved["path"]] = reference_pool.submit(
                                fetch_reference, saved
                            )
                self.log.event("batch_ready", phase=phase, job=self.job, step=step)
                training_start = time.time()
                losses, optimizer_metrics = [], []
                for offset in range(0, len(chosen), self.cfg["minibatch_groups"]):
                    mini = chosen[offset : offset + self.cfg["minibatch_groups"]]
                    pairs = [
                        (e, float(a))
                        for g in mini
                        for e, a in zip(
                            g["episodes"],
                            advantages([e["reward"] for e in g["episodes"]]),
                        )
                    ]
                    denominator = sum(e["generated_tokens"] for e, _ in pairs)
                    with ThreadPoolExecutor(16) as ref_pool:

                        def build(pair):
                            saved, a = pair
                            episode = load_episode(saved["path"])
                            refs = load_episode(
                                reference_futures[saved["path"]].result(timeout=3600)
                            )
                            return (
                                datum(
                                    episode, a, denominator, refs, self.cfg["kl_coef"]
                                ),
                                episode["mask"][1:],
                            )

                        built = list(ref_pool.map(build, pairs))
                        data, masks = map(list, zip(*built))
                    self.log.event(
                        "reference_ready",
                        phase=phase,
                        job=self.job,
                        step=step,
                        minibatch=offset // self.cfg["minibatch_groups"],
                    )
                    before = time.time()
                    fwd = self.trainer.forward_backward(
                        data,
                        "ppo",
                        loss_fn_config={
                            "clip_low_threshold": 0.8,
                            "clip_high_threshold": 1.28,
                        },
                    )
                    opt = self.trainer.optim_step(
                        types.AdamParams(learning_rate=1e-6, grad_clip_norm=1.0)
                    )
                    result = fwd.result(timeout=7200)
                    optimizer = opt.result(timeout=7200)
                    losses.append(result.metrics)
                    optimizer_metrics.append(optimizer.metrics)
                    difference = None
                    try:
                        difference = logprob_difference(
                            data, masks, result.loss_fn_outputs
                        )
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        self.log.event(
                            "logprob_difference_error",
                            phase=phase,
                            job=self.job,
                            step=step,
                            error=str(exc),
                        )
                    self.log.event(
                        "optimizer_step",
                        phase=phase,
                        job=self.job,
                        step=step,
                        minibatch=offset // self.cfg["minibatch_groups"],
                        request_s=time.time() - before,
                        assistant_tokens=denominator,
                        metrics=result.metrics,
                        optimizer_metrics=optimizer.metrics,
                        logprob_diff=difference,
                    )
                    if not all(
                        math.isfinite(float(x)) for x in result.metrics.values()
                    ):
                        raise InfrastructureError("Nonfinite training metrics")
                training_s = time.time() - training_start
                checkpoint_start = time.time()
                name = f"{phase}-job{self.job}-step{step + 1:04d}"
                state = self.trainer.save_state(name).result(timeout=3600)
                sampler = publish_policy(self.trainer, self.service, name)
                with self.lock:
                    self.sampler = sampler
                    self.version += 1
                row = self.log.event(
                    "step_done",
                    phase=phase,
                    job=self.job,
                    step=step,
                    incoming_reward_mean=float(np.mean(incoming)),
                    incoming_episodes=len(incoming),
                    selected_reward_mean=float(
                        np.mean([e["reward"] for g in chosen for e in g["episodes"]])
                    ),
                    groups=len(chosen),
                    generated_tokens=sum(
                        e["generated_tokens"] for g in chosen for e in g["episodes"]
                    ),
                    collection_s=training_start - started,
                    training_s=training_s,
                    checkpoint_s=time.time() - checkpoint_start,
                    total_s=time.time() - started,
                    checkpoint=state.path,
                    policy_version=self.version,
                )
                atomic_json(
                    self.log.root / f"{phase}-job{self.job}-checkpoint.json", row
                )
            # Last interval ends at the same boundary where the next step would start.
            self.log.event(
                "step_interval",
                phase=phase,
                job=self.job,
                step=self.cfg["steps"] - 1,
                start_to_start_s=time.time() - previous_start,
                terminal_boundary=True,
            )
        except Exception as exc:
            # Tell the supervisor before draining queued rollouts, which can
            # otherwise delay recovery from a dead trainer for many minutes.
            if engine_lost(exc) or isinstance(exc, TimeoutError):
                request_restart(self.log, self.job, exc)
            raise
        finally:
            stop.set()
            reference_pool.shutdown(wait=False, cancel_futures=True)
            producer.join()


def config(args):
    return dict(
        model=MODEL,
        context_tokens=262144,
        turn_tokens=8192,
        max_turns=20,
        rank=32,
        group_size=8,
        groups_per_batch=32,
        minibatch_groups=8,
        rollout_workers=args.rollout_workers,
        inflight_groups=args.inflight_groups,
        grading_workers=args.grading_workers,
        grading_queue_size=args.grading_queue_size,
        reference_token_reserve=16,
        completed_group_buffer=8,
        steps=args.steps,
        seed=4242,
        temperature=1.0,
        kl_coef=1e-4,
        policy_lag_max=1,
        hours=args.hours,
        trainer_gpus=4,
        trainer_gpu_options=["B300:4"],
        sampler_replicas=8,
        sampler_gpu_type="B300",
        benchmark_tasks=args.tasks,
        jobs=8,
        algorithm="async DAPO with frozen-reference KL reward shaping",
    )


def resume_records(source, destination, target_steps):
    """Prefer this continuation's checkpoints; otherwise use the prior run."""
    records = {}
    for job in range(8):
        name = f"eight-job{job}-checkpoint.json"
        path = destination / name if (destination / name).exists() else source / name
        record = json.loads(path.read_text())
        if record["job"] != job or record["phase"] != "eight":
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        if not record["checkpoint"].startswith("tinker://"):
            raise ValueError(f"Invalid checkpoint path: {path}")
        record["start_step"] = int(record["step"]) + 1
        if record["start_step"] > target_steps:
            raise ValueError("Target batch count precedes saved progress")
        record["start_group"] = 0
        records[job] = record
    for directory in {source, destination}:
        events = directory / "events.jsonl"
        if not events.exists():
            continue
        for line in events.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("phase") == "eight"
                and "group" in row
                and row.get("job") in records
            ):
                record = records[row["job"]]
                record["start_group"] = max(
                    record["start_group"], int(row["group"]) + 1
                )
    return records


def wait_for_client_start(job, anchor, spacing, cancelled):
    """Use a monotonic schedule; fail-fast cancellation wakes delayed clients."""
    delay = max(0.0, anchor + job * spacing - time.monotonic())
    if cancelled.wait(delay):
        raise RuntimeError("Experiment stopped before scheduled client start")


def run_staggered(service, tokenizer, env, tasks, cfg, log, reference, spacing):
    # Complete the shared cold start before starting the stagger clock. Later
    # clients create their adapters and immediately start rollouts, with no barrier.
    log.event("trainers_start", jobs=8)
    first = service.create_lora_training_client(
        MODEL,
        rank=32,
        train_unembed=False,
        user_metadata={"experiment": "swe-staggered", "job": "0"},
    )
    initial = first.save_state("initial").result(timeout=3600).path
    atomic_json(log.root / "initial.json", {"path": initial})
    first_sampler = publish_policy(first, service, "eight-job0-initial")
    anchor = time.monotonic()
    deadline = time.time() + cfg["hours"] * 3600
    cancelled = threading.Event()
    models = {}
    log.event(
        "phase_start",
        phase="eight",
        active_jobs=0,
        target_jobs=8,
        client_stagger_seconds=spacing,
    )

    def run(job):
        try:
            wait_for_client_start(job, anchor, spacing, cancelled)
            log.event(
                "client_start",
                phase="eight",
                job=job,
                scheduled_offset_s=job * spacing,
                actual_offset_s=time.monotonic() - anchor,
            )
            trainer = (
                first
                if job == 0
                else service.create_lora_training_client(
                    MODEL,
                    rank=32,
                    train_unembed=False,
                    user_metadata={"experiment": "swe-staggered", "job": str(job)},
                )
            )
            if job:
                trainer.load_state_with_optimizer(initial).result(timeout=3600)
            client = Job(
                job, trainer, service, tokenizer, env, tasks, cfg, log, reference
            )
            client.sampler = (
                first_sampler
                if job == 0
                else publish_policy(trainer, service, f"eight-job{job}-initial")
            )
            with log.lock:
                models[str(job)] = trainer.model_id
                atomic_json(
                    log.root / "trainers.json",
                    {"time": time.time(), "models": dict(models)},
                )
            log.event(
                "client_ready",
                phase="eight",
                job=job,
                model_id=trainer.model_id,
                actual_offset_s=time.monotonic() - anchor,
            )
            client.run("eight", deadline)
            log.event("client_done", phase="eight", job=job)
        except Exception as exc:
            cancelled.set()
            request_restart(log, job, exc)
            raise

    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(run, job) for job in range(8)]
        for future in futures:
            future.result()
    log.event("phase_end", phase="eight", active_jobs=8)
    log.event("completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--rollout-workers", type=int, default=64)
    parser.add_argument("--inflight-groups", type=int, default=32)
    parser.add_argument("--grading-workers", type=int, default=64)
    parser.add_argument("--grading-queue-size", type=int, default=64)
    parser.add_argument(
        "--stage",
        choices=["calibrate", "smoke", "benchmark", "train", "staggered"],
        default="benchmark",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--client-stagger-seconds", type=float, default=300)
    parser.add_argument("--base-url", default=URL)
    args = parser.parse_args()
    if (
        min(args.rollout_workers, args.inflight_groups, args.grading_workers) < 1
        or args.grading_queue_size < 0
    ):
        parser.error(
            "Worker and group limits must be positive; grading queue size must be nonnegative"
        )
    if args.steps < 1 or args.hours <= 0:
        parser.error("Training batches and hours must be positive")
    if args.stage == "train" and args.resume_from is None:
        parser.error("Training continuation requires --resume-from")
    if args.client_stagger_seconds < 0 or not math.isfinite(
        args.client_stagger_seconds
    ):
        parser.error("Client stagger must be finite and nonnegative")
    if args.stage != "calibrate" and not args.base_url:
        parser.error("Provide --base-url or SPINDLE_BASE_URL")
    cfg, log = config(args), Log(args.root)
    cfg["client_stagger_seconds"] = args.client_stagger_seconds
    atomic_json(args.root / "config.json", cfg)
    resumes = (
        resume_records(args.resume_from, args.root, args.steps)
        if args.stage == "train"
        else None
    )
    if resumes:
        atomic_json(args.root / "resume.json", {str(j): r for j, r in resumes.items()})
    tasks = json.loads((args.root / "train.json").read_text())
    random.Random(cfg["seed"]).shuffle(tasks)
    tasks = tasks[: args.tasks]
    atomic_json(args.root / "workload_ids.json", [t["instance_id"] for t in tasks])
    env = Environments(args.root)

    def calibrate():
        existing_pid = os.environ.get("SWE_CALIBRATION_PID")
        while existing_pid:
            try:
                os.kill(int(existing_pid), 0)
                time.sleep(5)
            except ProcessLookupError:
                break
        log.event("calibration_start", tasks=len(tasks))

        def validate(inst):
            rejected = (
                args.root / "calibration" / inst["instance_id"] / "unsupported.json"
            )
            if rejected.exists():
                return inst, None, json.loads(rejected.read_text())["reason"]
            try:
                return inst, env.calibrate(inst), None
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                rejected.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    rejected, {"instance_id": inst["instance_id"], "reason": reason}
                )
                return inst, None, reason

        valid = []
        with ThreadPoolExecutor(8) as pool:
            for inst, result, reason in pool.map(validate, tasks):
                if result is not None:
                    valid.append(inst)
                    log.event("calibration_pass", **result)
                else:
                    log.event(
                        "calibration_excluded",
                        instance_id=inst["instance_id"],
                        reason=reason,
                    )
        if len(valid) < min(16, args.tasks):
            raise InfrastructureError(
                "Too few environments passed both grader controls"
            )
        tasks[:] = valid
        cfg["validated_workload_tasks"] = len(tasks)
        atomic_json(args.root / "config.json", cfg)
        atomic_json(
            args.root / "validated_workload_ids.json", [t["instance_id"] for t in tasks]
        )

    if args.stage == "calibrate":
        calibrate()
        return
    tokenizer = AutoTokenizer.from_pretrained(str(args.root / "tokenizer"))
    service = tinker.ServiceClient(base_url=args.base_url, timeout=3600)
    reference = service.create_sampling_client(base_model=MODEL)
    if args.stage == "smoke":
        calibrate()
        episode = rollout(
            tasks[0],
            reference,
            tokenizer,
            env,
            cfg,
            log,
            dict(phase="smoke", job=0, group=0, attempt=0, policy_version=0),
        )
        log.event("smoke_complete", **episode)
        return
    if args.stage == "staggered":
        calibrate()
        run_staggered(
            service,
            tokenizer,
            env,
            tasks,
            cfg,
            log,
            reference,
            args.client_stagger_seconds,
        )
        return
    log.event("trainers_start", jobs=8)
    with ThreadPoolExecutor(8) as pool:
        trainers = list(
            pool.map(
                lambda j: service.create_lora_training_client(
                    MODEL,
                    rank=32,
                    train_unembed=False,
                    user_metadata={"experiment": "swe-lora-utilization", "job": str(j)},
                ),
                range(8),
            )
        )
    log.event(
        "trainers_ready", models={str(j): t.model_id for j, t in enumerate(trainers)}
    )
    if resumes:
        calibrate()
        expected = json.loads(
            (args.resume_from / "validated_workload_ids.json").read_text()
        )
        if [t["instance_id"] for t in tasks] != expected:
            raise ValueError("Continuation workload differs from checkpoint run")

        def restore(job):
            trainer, record = trainers[job], resumes[job]
            trainer.load_state_with_optimizer(record["checkpoint"]).result(timeout=3600)
            client = Job(
                job, trainer, service, tokenizer, env, tasks, cfg, log, reference
            )
            client.sampler = publish_policy(
                trainer, service, f"resume-job{job}-step{record['start_step']:04d}"
            )
            log.event(
                "checkpoint_restored",
                job=job,
                checkpoint=record["checkpoint"],
                completed_batches=record["start_step"],
                optimizer_restored=True,
            )
            return client

        with ThreadPoolExecutor(8) as pool:
            jobs = list(pool.map(restore, range(8)))
        log.event(
            "phase_start",
            phase="eight",
            active_jobs=8,
            continuation=True,
            target_batches_per_job=args.steps,
        )
        deadline = time.time() + args.hours * 3600

        def train_to_target(job):
            record = resumes[job.job]
            failures, last_checkpoint = 0, record["checkpoint"]
            while True:
                try:
                    job.run(
                        "eight", deadline, record["start_step"], record["start_group"]
                    )
                    return
                except Exception as exc:
                    if engine_lost(exc) or isinstance(exc, TimeoutError):
                        request_restart(log, job.job, exc)
                        raise
                    record = resume_records(args.resume_from, args.root, args.steps)[
                        job.job
                    ]
                    if record["checkpoint"] != last_checkpoint:
                        failures = 0
                        last_checkpoint = record["checkpoint"]
                    failures += 1
                    if failures >= 3 or time.time() >= deadline:
                        log.event(
                            "job_failed",
                            job=job.job,
                            checkpoint=last_checkpoint,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        request_restart(log, job.job, exc)
                        raise
                    log.event(
                        "job_retry",
                        job=job.job,
                        checkpoint=last_checkpoint,
                        attempt=failures,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    time.sleep(30)
                    try:
                        job.trainer.load_state_with_optimizer(last_checkpoint).result(
                            timeout=3600
                        )
                        job.sampler = publish_policy(
                            job.trainer,
                            service,
                            f"retry-job{job.job}-step{record['start_step']:04d}-{time.time_ns()}",
                        )
                    except Exception as restore_error:
                        request_restart(log, job.job, restore_error)
                        raise

        with ThreadPoolExecutor(8) as pool:
            futures = [pool.submit(train_to_target, job) for job in jobs]
            for future in futures:
                future.result()
        log.event("phase_end", phase="eight", active_jobs=8)
        log.event("completed")
        return
    initial = trainers[0].save_state("initial").result(timeout=3600).path
    atomic_json(args.root / "initial.json", {"path": initial})
    jobs = [
        Job(j, trainer, service, tokenizer, env, tasks, cfg, log, reference)
        for j, trainer in enumerate(trainers)
    ]
    log.event("warmup_start")

    def warm(job):
        job.sampler = publish_policy(job.trainer, service, f"warm-before-{job.job}")
        prompt = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Compute 13 + 29. Answer briefly."}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        sample = (
            job.sampler.sample(
                types.ModelInput.from_ints(prompt),
                1,
                types.SamplingParams(max_tokens=128, temperature=1.0),
            )
            .result(timeout=3600)
            .sequences[0]
        )
        tokens = prompt + list(sample.tokens)
        episode = {
            "tokens": tokens,
            "mask": [0] * len(prompt) + [1] * len(sample.tokens),
            "behavior_logprobs": [0.0] * len(prompt) + list(sample.logprobs),
        }
        refs = reference.compute_logprobs(types.ModelInput.from_ints(tokens)).result(
            timeout=3600
        )
        data = [datum(episode, 1.0, len(sample.tokens), refs, cfg["kl_coef"])]
        fwd = job.trainer.forward_backward(
            data,
            "ppo",
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
        )
        opt = job.trainer.optim_step(
            types.AdamParams(learning_rate=1e-6, grad_clip_norm=1.0)
        )
        metrics = fwd.result(timeout=7200).metrics
        opt.result(timeout=7200)
        if not all(math.isfinite(float(v)) for v in metrics.values()):
            raise InfrastructureError("Nonfinite warmup loss")
        job.trainer.load_state_with_optimizer(initial).result(timeout=3600)
        job.sampler = publish_policy(job.trainer, service, f"warm-after-{job.job}")
        log.event(
            "warmup_job_pass",
            job=job.job,
            metrics=metrics,
            note="Synthetic gradient smoke test; initial weights and optimizer restored",
        )

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(warm, jobs))
    log.event("warmup_end")
    calibrate()
    deadline = time.time() + args.hours * 3600
    log.event("phase_start", phase="single", active_jobs=1)
    jobs[0].run("single", deadline)
    log.event("phase_end", phase="single", active_jobs=1)
    trainers[0].load_state_with_optimizer(initial).result(timeout=3600)
    jobs[0].sampler = publish_policy(trainers[0], service, "eight-job0-reset")
    log.event("phase_start", phase="eight", active_jobs=8)
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(job.run, "eight", deadline) for job in jobs]
        for future in futures:
            future.result()
    log.event("phase_end", phase="eight", active_jobs=8)
    log.event("completed")


if __name__ == "__main__":
    main()
