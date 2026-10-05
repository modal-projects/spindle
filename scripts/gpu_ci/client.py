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

import numpy as np
import tinker
from tinker import types
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent


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
    if cfg["model"] != "Qwen/Qwen3.5-9B":
        raise ValueError("This experiment requires the non-Base Qwen3.5-9B model")
    if cfg["updates"] < 1 or cfg["max_tokens"] >= cfg["context_tokens"]:
        raise ValueError("Invalid update count or context budget")
    return cfg


class Client:
    def __init__(self, cfg, root, base_url, start_file=None):
        self.cfg, self.root = cfg, root
        self.start_file = start_file
        root.mkdir(parents=True, exist_ok=False)
        self.service = tinker.ServiceClient(
            **({"base_url": base_url} if base_url else {})
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"], revision=cfg["tokenizer_revision"]
        )
        self.rows = [
            json.loads(line) for line in (ROOT / "dapo.jsonl").read_text().splitlines()
        ]
        self.prompts = [
            list(
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": row["prompt"]
                            + "\nGive the final answer in \\boxed{}.",
                        }
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
            dataset_sha256=hashlib.sha256(
                (ROOT / "dapo.jsonl").read_bytes()
            ).hexdigest(),
            initialization_seed=None,
            stop_token_ids=self.stops,
            base_url=base_url or "Tinker SDK default",
            packages={
                n: importlib.metadata.version(n)
                for n in ["tinker", "transformers", "numpy"]
            },
        )
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2))
        self.event("client_create_start")
        self.trainer = create_trainer(self.service, cfg["model"], cfg["rank"])
        self.event("client_created", model_id=self.trainer.model_id)
        self.next_group = 0

    def event(self, event, **fields):
        row = dict(event=event, time=time.time(), **fields)
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def publish(self, update, future=None):
        start = time.time()
        if future is None:
            future = self.trainer.save_weights_for_sampler(f"policy-{update:03d}")
        path = future.result(timeout=1800).path
        run_id = (
            self.trainer.model_id
            if ":train:" in self.trainer.model_id
            else self.trainer.model_id + ":train:0"
        )
        if not path.startswith(f"tinker://{run_id}/sampler_weights/"):
            raise ValueError("Published sampler belongs to a different model")
        saved = time.time()
        self.event("publish_weights_received", update=update, seconds=saved - start)
        sampler = self.service.create_sampling_client(model_path=path)
        self.event(
            "publish",
            update=update,
            path=path,
            seconds=time.time() - start,
            weights_seconds=saved - start,
            sampling_client_seconds=time.time() - saved,
        )
        return sampler, update

    def group(self, index, snapshot):
        sampler, version = snapshot
        cfg = self.cfg
        # All eight independent clients use the same task/seed schedule as the
        # single-client Tinker comparison. Client identity never changes inputs.
        row = self.rows[index % len(self.rows)]
        prompt = self.prompts[index % len(self.prompts)]
        start = time.time()
        result = sampler.sample(
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
        ).result(timeout=1800)
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
                    reward=float(
                        answer(text) is not None
                        and answer(text) == answer(str(row["answer"]))
                    ),
                    stop_reason=str(seq.stop_reason),
                )
            )
        saved = dict(
            group=index,
            row=index % len(self.rows),
            policy_version=version,
            prompt=prompt,
            start=start,
            end=time.time(),
            sequences=sequences,
        )
        with gzip.open(self.root / f"group-{index:06d}.json.gz", "wt") as f:
            json.dump(saved, f)
        return saved

    def rollout(self, snapshot, update):
        cfg = self.cfg
        start = time.time()
        kept = []
        all_groups = []
        with ThreadPoolExecutor(self.cfg["groups_per_update"]) as pool:
            while len(kept) < self.cfg["groups_per_update"]:
                count = min(
                    self.cfg["groups_per_update"] - len(kept),
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
                kept.extend(
                    g
                    for g in groups
                    if not cfg.get("dynamic_group_filtering", True)
                    or len({s["reward"] for s in g["sequences"]}) > 1
                )
        lengths = [len(s["tokens"]) for g in all_groups for s in g["sequences"]]
        generated = sum(lengths)
        trained = sum(len(s["tokens"]) for g in kept for s in g["sequences"])
        data = []
        for group in kept:
            rewards = np.array([s["reward"] for s in group["sequences"]])
            advantages = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)
            data.extend(
                datum(group["prompt"], s["tokens"], s["logprobs"], float(a), trained)
                for s, a in zip(group["sequences"], advantages, strict=True)
            )
        stats = dict(
            update=update,
            policy_version=snapshot[1],
            seconds=time.time() - start,
            groups_sampled=len(all_groups),
            groups_filtered=len(all_groups) - len(kept),
            output_tokens=generated,
            used_output_tokens=trained,
            sampling_input_tokens=sum(
                len(g["prompt"]) * len(g["sequences"]) for g in all_groups
            ),
            training_input_tokens=sum(len(d.model_input.to_ints()) for d in data),
            output_lengths=lengths,
            truncated=sum(n >= self.cfg["max_tokens"] for n in lengths),
            reward=float(
                np.mean([s["reward"] for g in all_groups for s in g["sequences"]])
            ),
            mixed_groups=sum(
                len({s["reward"] for s in g["sequences"]}) > 1 for g in kept
            ),
        )
        self.event("rollout_done", **stats)
        return data, stats

    def run(self):
        cfg = self.cfg
        snapshot = self.publish(0)
        # Finish a real sampling request before starting the stagger clock so
        # the initial shared inference cold start does not synchronize clients.
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
        self.probe_tokens = self.prompts[0] + list(warm.sequences[0].tokens)
        initial_probe = (
            snapshot[0]
            .compute_logprobs(types.ModelInput.from_ints(self.probe_tokens))
            .result(timeout=1800)
        )
        self.event("ready")
        if self.start_file is not None:
            deadline = time.monotonic() + 1800
            while not self.start_file.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Other CI clients did not become ready")
                time.sleep(1)
        self.event("training_started")
        started = time.time()
        with ThreadPoolExecutor(1) as prefetch:
            pending = prefetch.submit(self.rollout, snapshot, 1)
            for update in range(1, cfg["updates"] + 1):
                begin = time.time()
                data, stats = pending.result()
                lag = update - 1 - stats["policy_version"]
                if not 0 <= lag <= 1:
                    raise RuntimeError(f"Unexpected policy lag {lag}")
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
                publication = (
                    self.trainer.save_weights_for_sampler(f"policy-{update:03d}")
                    if update < cfg["updates"]
                    else None
                )
                if publication is not None:
                    self.event("publish_enqueued", update=update)
                result = forward.result(timeout=1800)
                opt = optimizer.result(timeout=1800)
                if not all(
                    math.isfinite(float(v))
                    for v in [*result.metrics.values(), *opt.metrics.values()]
                ):
                    raise RuntimeError("Nonfinite training metrics")
                if opt.metrics.get("update_successful:mean") != 1.0:
                    raise RuntimeError("Optimizer update did not report success")
                self.event(
                    "train_done",
                    update=update,
                    seconds=time.time() - train_start,
                    metrics=result.metrics,
                    optimizer_metrics=opt.metrics,
                )
                if update < cfg["updates"]:
                    snapshot = self.publish(update, publication)
                self.event(
                    "update_done",
                    update=update,
                    seconds=time.time() - begin,
                    elapsed_seconds=time.time() - started,
                    policy_lag=lag,
                )
        self.event(
            "training_finished", updates=cfg["updates"], seconds=time.time() - started
        )
        final_sampler, _ = self.publish(cfg["updates"])
        final_probe = final_sampler.compute_logprobs(
            types.ModelInput.from_ints(self.probe_tokens)
        ).result(timeout=1800)
        if len(initial_probe) != len(final_probe):
            raise RuntimeError("Probe logprob lengths changed")
        if any((a is None) != (b is None) for a, b in zip(initial_probe, final_probe)):
            raise RuntimeError("Probe logprob alignment changed")
        pairs = [
            (a, b)
            for a, b in zip(initial_probe, final_probe)
            if a is not None and b is not None
        ]
        if not pairs or not all(math.isfinite(v) for pair in pairs for v in pair):
            raise RuntimeError("Invalid probe logprobs")
        self.event("policy_probe", max_logprob_change=max(abs(a - b) for a, b in pairs))
        self.event("completed", updates=cfg["updates"], seconds=time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--start-file", type=Path)
    args = parser.parse_args()
    client = Client(
        read_config(args.config), args.output, args.base_url, args.start_file
    )
    try:
        client.run()
    except Exception as exc:
        client.event("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
