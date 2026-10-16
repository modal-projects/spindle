"""Identical Tinker SDK training loop for either endpoint; no Modal dependency."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

from scripts.benchmarks.lora_routing.recovery import atomic_json, latest_commit

import httpx
import numpy as np
import tinker
from tinker import types
from transformers import AutoTokenizer
from scripts.benchmarks.lora_routing.sdk_affinity import PromptAffinityTransport
from scripts.benchmarks.rl_rust.codeforces import initial_messages, sample_group

ROOT = Path(__file__).resolve().parent


class RolloutExecutor(ThreadPoolExecutor):
    def __exit__(self, kind, value, tb):
        self.shutdown(wait=kind is None, cancel_futures=kind is not None)
        return False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def answer(text):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    text = matches[-1] if matches else text.strip()
    try:
        result = Decimal(text.replace(",", "").strip())
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def datum(prompt, tokens, logprobs, advantage, denominator):
    if (
        not tokens
        or len(tokens) != len(logprobs)
        or not all(map(math.isfinite, logprobs))
    ):
        raise ValueError("Missing or invalid behavior tokens/logprobs")
    prefix = len(prompt) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints((prompt + tokens)[:-1]),
        loss_fn_inputs={
            "target_tokens": (prompt + tokens)[1:],
            "logprobs": [0.0] * prefix + logprobs,
            "advantages": [0.0] * prefix + [advantage / denominator] * len(tokens),
        },
    )


def create_trainer(service, model, rank):
    # Miles rejects per-adapter seeds. Use this same call on both endpoints.
    return service.create_lora_training_client(
        model, rank=rank, train_attn=True, train_mlp=True, train_unembed=False
    )


def read_config(path):
    cfg = json.loads(path.read_text())
    if cfg["model"] not in {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"}:
        raise ValueError("This benchmark supports Qwen3.5-9B and Qwen3.8-27B")
    if cfg["updates"] < 1 or cfg["max_tokens"] >= cfg["context_tokens"]:
        raise ValueError("Invalid update count or context budget")
    return cfg


class Client:
    def __init__(self, cfg, root, base_url, client_index=0):
        self.cfg, self.root = cfg, root
        self.client_index = client_index
        self.base_url = base_url
        root.mkdir(parents=True, exist_ok=True)
        self.recover = cfg.get("checkpoint_each_update", False)
        if self.recover and cfg.get("dynamic_group_filtering", True):
            raise ValueError(
                "Checkpoint resume requires fixed unfiltered group indices"
            )
        self.commit = latest_commit(root) if self.recover else None
        self.completed_update = self.commit["update"] if self.commit else 0
        self.sampling_paths = {}
        self.attempt = len(list(root.glob("attempt-*.json")))
        atomic_json(
            root / f"attempt-{self.attempt:03d}.json", dict(started=time.time())
        )
        options = {"base_url": base_url} if base_url else {}
        if cfg.get("prompt_affinity", False):
            if not base_url:
                raise ValueError("Prompt affinity is a Spindle API extension")
            options["http_client"] = httpx.AsyncClient(
                transport=PromptAffinityTransport(), timeout=300, trust_env=False
            )
        self.service = tinker.ServiceClient(**options)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"], revision=cfg["tokenizer_revision"]
        )
        codeforces = cfg.get("task") == "codeforces"
        dataset = Path(cfg["dataset"]) if codeforces else ROOT / "dapo.jsonl"
        self.rows = (
            json.loads(dataset.read_text())["problems"]
            if codeforces
            else [json.loads(line) for line in dataset.read_text().splitlines()]
        )
        self.prompts = [
            list(
                self.tokenizer.apply_chat_template(
                    initial_messages(row)
                    if codeforces
                    else [
                        dict(
                            role="user",
                            content=row["prompt"]
                            + cfg.get(
                                "prompt_suffix", "\nGive the final answer in \\boxed{}."
                            ),
                        )
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=cfg["thinking"],
                    return_dict=False,
                )
            )
            for row in self.rows
        ]
        if any(
            len(p) + cfg["max_tokens"] >= cfg["context_tokens"] for p in self.prompts
        ):
            raise ValueError("A prompt exceeds the configured context budget")
        self.stops = [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.eos_token_id,
        ]
        self.stops = list(dict.fromkeys(self.stops))
        metadata = dict(
            config=cfg,
            config_sha256=digest(cfg),
            prompts_sha256=digest(self.prompts),
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            initialization_seed=None,
            client_index=client_index,
            problem_offset=client_index * cfg.get("client_problem_stride", 0),
            stop_token_ids=self.stops,
            base_url=base_url or "Tinker SDK default",
            packages={
                n: importlib.metadata.version(n)
                for n in ["tinker", "transformers", "numpy"]
            },
        )
        metadata["source_sha256"] = {
            "codeforces.py": hashlib.sha256(
                (ROOT.parent / "rl_rust/codeforces.py").read_bytes()
            ).hexdigest()
        }
        for name in ("client.py", "sdk_affinity.py", "recovery.py"):
            content = (ROOT / name).read_bytes()
            metadata["source_sha256"][name] = hashlib.sha256(content).hexdigest()
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        manifest = root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != metadata:
                raise ValueError("Resume manifest differs from original inputs/source")
        else:
            atomic_json(manifest, metadata)
        self.event("client_create_start")
        self.trainer = create_trainer(self.service, cfg["model"], cfg["rank"])
        self.event("client_created", model_id=self.trainer.model_id)
        if self.commit:
            began = time.time()
            self.trainer.load_state_with_optimizer(
                self.commit["checkpoint_path"]
            ).result(timeout=cfg.get("training_timeout_seconds", 7200))
            self.event(
                "checkpoint_restored",
                update=self.completed_update,
                path=self.commit["checkpoint_path"],
                seconds=time.time() - began,
            )
        elif cfg.get("initial_state_paths"):
            source = cfg["initial_state_paths"][client_index]
            began = time.time()
            self.trainer.load_state(source).result(timeout=1800)
            self.event("initial_state_loaded", path=source, seconds=time.time() - began)
        if cfg.get("save_initial_state", False):
            began = time.time()
            checkpoint = self.trainer.save_state("routing-initial").result(timeout=1800)
            self.initial_state_path = checkpoint.path
            self.event(
                "initial_state_saved", path=checkpoint.path, seconds=time.time() - began
            )
        self.next_group = self.completed_update * cfg["groups_per_update"]

    def event(self, event, **fields):
        row = dict(event=event, time=time.time(), attempt=self.attempt, **fields)
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def publish(self, update, future=None):
        start = time.time()
        if future is None:
            future = self.trainer.save_weights_for_sampler(f"policy-{update:03d}")
        path = future.result(
            timeout=self.cfg.get("training_timeout_seconds", 7200)
        ).path
        saved = time.time()
        self.event("publish_weights_received", update=update, seconds=saved - start)
        self.sampling_paths[update] = path
        sampler = self.service.create_sampling_client(model_path=path)
        self.event(
            "publish",
            update=update,
            model_path=path,
            seconds=time.time() - start,
            weights_seconds=saved - start,
            sampling_client_seconds=time.time() - saved,
        )
        return sampler, update

    def group(self, index, snapshot):
        sampler, version = snapshot
        cached = self.root / f"group-{index:06d}.json.gz"
        if self.recover and cached.exists():
            with gzip.open(cached, "rt") as f:
                saved = json.load(f)
            if saved["policy_version"] != version or saved["group"] != index:
                raise ValueError("Cached rollout policy/group mismatch")
            self.event("group_reused", group=index, policy_version=version)
            return saved
        cfg = self.cfg
        # The optional offset gives independent jobs different problem streams;
        # client indices and offsets are identical across paired routing arms.
        row_index = (
            index + self.client_index * cfg.get("client_problem_stride", 0)
        ) % len(self.rows)
        if cfg.get("task") == "codeforces":
            saved = sample_group(self, index, snapshot, row_index)
            self.save_group(saved)
            return saved
        row = self.rows[row_index]
        prompt = self.prompts[row_index]
        start = time.time()
        future = sampler.sample(
            types.ModelInput.from_ints(prompt),
            cfg["samples_per_group"],
            types.SamplingParams(
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                stop=self.stops,
                seed=cfg["seed"] + index,
            ),
        )
        self.event("group_submitted", group=index, policy_version=version)
        result = future.result(timeout=3600)
        if len(result.sequences) != cfg["samples_per_group"]:
            raise ValueError("Incomplete sample group")
        sequences = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs or [])
            if (
                not tokens
                or len(tokens) != len(logprobs)
                or not all(map(math.isfinite, logprobs))
            ):
                raise ValueError("Invalid sampled token/logprob alignment")
            text = self.tokenizer.decode(tokens)
            sequences.append(
                dict(
                    tokens=tokens,
                    logprobs=logprobs,
                    text=text,
                    reward=float(answer(text) == answer(str(row["answer"]))),
                    stop_reason=str(seq.stop_reason),
                )
            )
        saved = dict(
            group=index,
            row=row_index,
            policy_version=version,
            prompt=prompt,
            start=start,
            end=time.time(),
            first_sequence_cached_tokens=result.prompt_cache_hit_tokens,
            sequences=sequences,
        )
        self.save_group(saved)
        return saved

    def save_group(self, saved):
        index = saved["group"]
        temporary = self.root / f"group-{index:06d}.json.gz.tmp"
        with gzip.open(temporary, "wt") as f:
            json.dump(saved, f)
        temporary.replace(self.root / f"group-{index:06d}.json.gz")
        self.event(
            "group_completed",
            group=index,
            policy_version=saved["policy_version"],
            seconds=saved["end"] - saved["start"],
            output_tokens=sum(len(s["tokens"]) for s in saved["sequences"]),
        )

    def rollout(self, snapshot, update):
        cfg = self.cfg
        start = time.time()
        if self.recover:
            intent = self.root / f"rollout-{update:03d}.json"
            if intent.exists():
                previous = json.loads(intent.read_text())
                start = previous["started_at"]
                if previous["policy_version"] != snapshot[1]:
                    raise ValueError("Resume changed behavior policy")
            else:
                atomic_json(
                    intent,
                    dict(
                        started_at=start,
                        policy_version=snapshot[1],
                        model_path=self.sampling_paths[snapshot[1]],
                    ),
                )
        kept = []
        all_groups = []
        filtered = unused = 0
        self.event("rollout_start", update=update, policy_version=snapshot[1])
        with RolloutExecutor(self.cfg["groups_per_update"]) as pool:
            while len(kept) < self.cfg["groups_per_update"]:
                count = min(
                    max(
                        self.cfg["groups_per_update"] - len(kept),
                        cfg.get("minimum_refill_groups", 1),
                    ),
                    self.cfg["max_candidate_groups"] - len(all_groups),
                )
                if count <= 0:
                    raise RuntimeError(
                        "Too few mixed-reward groups within candidate budget"
                    )
                indices = list(range(self.next_group, self.next_group + count))
                self.next_group += count
                groups = list(pool.map(lambda i: self.group(i, snapshot), indices))
                all_groups.extend(groups)
                eligible = [
                    g
                    for g in groups
                    if not cfg.get("dynamic_group_filtering", True)
                    or len({s["reward"] for s in g["sequences"]}) > 1
                ]
                needed = cfg["groups_per_update"] - len(kept)
                kept.extend(eligible[:needed])
                filtered += len(groups) - len(eligible)
                unused += max(0, len(eligible) - needed)
        lengths = [len(s["tokens"]) for g in all_groups for s in g["sequences"]]
        generated = sum(lengths)
        trained = sum(len(s["tokens"]) for g in kept for s in g["sequences"])
        data = []
        for group in kept:
            rewards = np.array([s["reward"] for s in group["sequences"]])
            advantages = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)
            for sequence, advantage in zip(group["sequences"], advantages, strict=True):
                for turn in sequence.get(
                    "turns", [dict(prompt=group["prompt"], **sequence)]
                ):
                    data.append(
                        datum(
                            turn["prompt"],
                            turn["tokens"],
                            turn["logprobs"],
                            float(advantage),
                            trained,
                        )
                    )
        stats = dict(
            update=update,
            policy_version=snapshot[1],
            started_at=start,
            seconds=time.time() - start,
            groups_sampled=len(all_groups),
            groups_filtered=filtered,
            groups_unused=unused,
            groups_retained=len(kept),
            mixed_reward_groups=sum(
                len({s["reward"] for s in g["sequences"]}) > 1 for g in kept
            ),
            output_tokens=generated,
            used_output_tokens=trained,
            sampling_input_tokens=sum(
                len(turn["prompt"])
                for g in all_groups
                for seq in g["sequences"]
                for turn in seq.get("turns", [dict(prompt=g["prompt"])])
            ),
            training_input_tokens=sum(len(d.model_input.to_ints()) for d in data),
            output_lengths=lengths,
            truncated=sum(n >= self.cfg["max_tokens"] for n in lengths),
            reward=float(
                np.mean([s["reward"] for g in all_groups for s in g["sequences"]])
            ),
        )
        self.event("rollout_done", **stats)
        return data, stats

    def run(self):
        cfg = self.cfg
        if self.completed_update == cfg["updates"]:
            self.event("completed", updates=cfg["updates"], recovered=True)
            self.unload()
            return
        if self.commit:
            version = self.completed_update
            self.sampling_paths[version] = self.commit["published_path"]
            snapshot = (
                self.service.create_sampling_client(
                    model_path=self.commit["published_path"]
                ),
                version,
            )
            # Preserve the original one-update-lagged prefetched behavior policy.
            version, path = (
                self.commit["next_policy_version"],
                self.commit["next_policy_path"],
            )
            self.sampling_paths[version] = path
            first_snapshot = (
                self.service.create_sampling_client(model_path=path),
                version,
            )
        else:
            snapshot = self.publish(0)
            first_snapshot = snapshot
        # Large-load comparisons prewarm the pools before launching clients.
        # A per-client warmup otherwise blocks real rollout submission behind
        # other clients' long generations and changes the intended load shape.
        if cfg["warmup_samples"]:
            warm = (
                snapshot[0]
                .sample(
                    types.ModelInput.from_ints(self.prompts[0]),
                    cfg["warmup_samples"],
                    types.SamplingParams(
                        max_tokens=cfg["warmup_max_tokens"],
                        temperature=cfg["temperature"],
                        top_p=cfg["top_p"],
                        top_k=cfg["top_k"],
                        stop=self.stops,
                        seed=cfg["seed"],
                    ),
                )
                .result(timeout=1800)
            )
            self.event(
                "sampling_warmup",
                output_tokens=sum(len(s.tokens) for s in warm.sequences),
                sampling_input_tokens=len(self.prompts[0]) * len(warm.sequences),
            )
        self.event("ready")
        started = time.time()
        with RolloutExecutor(1) as prefetch:
            pending = prefetch.submit(
                self.rollout, first_snapshot, self.completed_update + 1
            )
            for update in range(self.completed_update + 1, cfg["updates"] + 1):
                begin = time.time()
                data, stats = pending.result()
                lag = update - 1 - stats["policy_version"]
                if not 0 <= lag <= 1:
                    raise RuntimeError(f"Unexpected policy lag {lag}")
                next_policy_version = snapshot[1]
                next_policy_path = self.sampling_paths[snapshot[1]]
                if update < cfg["updates"]:
                    pending = prefetch.submit(self.rollout, snapshot, update + 1)
                train_start = time.time()
                self.event("train_start", update=update, policy_lag=lag)
                forward = self.trainer.forward_backward(
                    data,
                    "ppo",
                    loss_fn_config={
                        "clip_low_threshold": cfg["clip_low"],
                        "clip_high_threshold": cfg["clip_high"],
                    },
                )
                optimizer = self.trainer.optim_step(
                    types.AdamParams(
                        learning_rate=cfg["learning_rate"],
                        beta1=cfg["beta1"],
                        beta2=cfg["beta2"],
                        eps=cfg["eps"],
                        weight_decay=0.0,
                        grad_clip_norm=cfg["grad_clip_norm"],
                    )
                )
                # Queue the snapshot behind the optimizer before awaiting HTTP
                # results. Server sequence ordering preserves the exact policy.
                publication = self.trainer.save_weights_for_sampler(
                    f"policy-{update:03d}"
                )
                if publication is not None:
                    self.event("publish_enqueued", update=update)
                result = forward.result(
                    timeout=cfg.get("training_timeout_seconds", 7200)
                )
                opt = optimizer.result(
                    timeout=cfg.get("training_timeout_seconds", 7200)
                )
                if not all(
                    math.isfinite(float(v))
                    for metrics in (result.metrics, opt.metrics)
                    for v in metrics.values()
                ):
                    raise RuntimeError("Nonfinite training or optimizer metrics")
                if opt.metrics.get("update_successful:mean", 1) != 1:
                    raise RuntimeError("Optimizer update was skipped")
                train_record = dict(
                    event="train_done",
                    time=time.time(),
                    update=update,
                    seconds=time.time() - train_start,
                    metrics=result.metrics,
                    optimizer_metrics=opt.metrics,
                )
                self.event(**{k: v for k, v in train_record.items() if k != "time"})
                snapshot = self.publish(update, publication)
                publication_time = time.time()
                checkpoint_seconds = 0.0
                if self.recover:
                    checkpoint_started = time.time()
                    checkpoint = self.trainer.save_state(f"resume-{update:03d}").result(
                        timeout=cfg.get("training_timeout_seconds", 7200)
                    )
                    checkpoint_seconds = time.time() - checkpoint_started
                done = dict(
                    update=update,
                    seconds=publication_time - stats["started_at"],
                    wait_and_train_seconds=publication_time - begin,
                    elapsed_seconds=publication_time - started,
                    policy_lag=lag,
                    publication_time=publication_time,
                    checkpoint_seconds=checkpoint_seconds,
                )
                if self.recover:
                    atomic_json(
                        self.root / f"commit-{update:03d}.json",
                        dict(
                            update=update,
                            checkpoint_path=checkpoint.path,
                            published_path=self.sampling_paths[update],
                            next_policy_version=next_policy_version,
                            next_policy_path=next_policy_path,
                            rollout=dict(event="rollout_done", **stats),
                            train=train_record,
                            done=dict(event="update_done", time=time.time(), **done),
                        ),
                    )
                self.event("update_done", **done)
        self.event("completed", updates=cfg["updates"], seconds=time.time() - started)
        self.unload()

    def unload(self):
        if self.base_url:
            # The supervisor owns final deployment cleanup. A timed-out unload
            # may still finish on the engine; it must not abort other clients'
            # valid training after this client has completed all its updates.
            try:
                response = httpx.post(
                    self.base_url + "/api/v1/unload_model",
                    headers={"x-api-key": os.environ["TINKER_API_KEY"]},
                    json={"model_id": self.trainer.model_id},
                    timeout=120,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                self.event("cleanup_warning", error=str(exc))
                return False
            else:
                self.event("unloaded")
                return True
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--client-index", type=int, default=0)
    args = parser.parse_args()
    client = None
    try:
        client = Client(
            read_config(args.config), args.output, args.base_url, args.client_index
        )
        client.run()
    except Exception as exc:
        failure = dict(
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            time=time.time(),
        )
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "failure.json", failure)
        if client is not None:
            client.event("failed", error=failure["error"])
        # A rollout executor may still be waiting on a dead backend. The supervisor
        # owns process-group cleanup; don't block failure reporting on those threads.
        raise


if __name__ == "__main__":
    main()
