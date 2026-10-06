"""Replay archived DAPO updates through an unchanged public Tinker training API.

No sampling, publication, or checkpointing. Each client has its own adapter.
First two replay updates are warmup; every tensor and expected backend chunk
hash is recorded before the timed call. Never write API keys to results.
"""

import argparse
import gzip
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker.lib._pydantic_conv import to_pydantic_input


def batches(root, client, update):
    groups = []
    for group in range((update - 1) * 8, update * 8):
        path = root / f"client-{client:02d}" / f"group-{group:06d}.json.gz"
        with gzip.open(path, "rt") as f:
            groups.append(json.load(f))
    denominator = sum(len(s["tokens"]) for g in groups for s in g["sequences"])
    data = []
    for group in groups:
        prompt = group["prompt"]
        rewards = np.array([s["reward"] for s in group["sequences"]])
        advantages = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)
        for seq, advantage in zip(group["sequences"], advantages, strict=True):
            tokens = seq["tokens"]
            prefix = len(prompt) - 1
            data.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints((prompt + tokens)[:-1]),
                    loss_fn_inputs={
                        "target_tokens": (prompt + tokens)[1:],
                        "logprobs": [0.0] * prefix + seq["logprobs"],
                        "advantages": [0.0] * prefix
                        + [float(advantage) / denominator] * len(tokens),
                    },
                )
            )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--start-update", type=int, default=8)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--startup-stagger", type=float, default=0.0)
    parser.add_argument("--submission-stagger", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(
            {**vars(args), "archive": str(args.archive), "output": str(args.output)},
            indent=2,
        )
    )
    barrier = threading.Barrier(args.clients, timeout=1800)

    def client(index):
        path = args.output / f"client-{index:02d}.jsonl"

        def event(name, **fields):
            with path.open("a") as f:
                f.write(
                    json.dumps(
                        {"event": name, "time": time.time(), "client": index, **fields},
                        allow_nan=False,
                    )
                    + "\n"
                )

        try:
            time.sleep(index * args.startup_stagger)
            event("create_start")
            service = tinker.ServiceClient(base_url=args.base_url)
            trainer = service.create_lora_training_client(
                "Qwen/Qwen3.5-9B",
                rank=32,
                train_attn=True,
                train_mlp=True,
                train_unembed=False,
            )
            event("ready", model_id=trainer.model_id)
            barrier.wait()
            config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
            for step in range(args.updates):
                update = args.start_update + step
                data = batches(args.archive, index, update)
                hashes = []
                for chunk in trainer._chunked_requests_generator(data):
                    payload = types.ForwardBackwardInput(
                        data=chunk, loss_fn="ppo", loss_fn_config=config
                    )
                    raw = to_pydantic_input(payload).model_dump(
                        mode="json", exclude_defaults=True
                    )
                    hashes.append(
                        hashlib.sha256(
                            json.dumps(
                                raw,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode()
                        ).hexdigest()
                    )
                start_seq = trainer._request_id_counter + 1
                event(
                    "prepared",
                    step=step,
                    archive_update=update,
                    model_id=trainer.model_id,
                    sha256=hashes,
                    first_seq=start_seq,
                    chunks=len(hashes),
                    input_tokens=sum(d.model_input.length for d in data),
                    examples=len(data),
                )
                if step == 0:
                    barrier.wait()
                    time.sleep(index * args.submission_stagger)
                start = time.time()
                event("forward_start", step=step)
                future = trainer.forward_backward(data, "ppo", loss_fn_config=config)
                optimizer = trainer.optim_step(
                    types.AdamParams(
                        learning_rate=1e-5,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-12,
                        weight_decay=0.0,
                        grad_clip_norm=1.0,
                    )
                )
                result = future.result(timeout=1800)
                if not all(math.isfinite(float(v)) for v in result.metrics.values()):
                    raise ValueError("nonfinite training metrics")
                event(
                    "forward_done",
                    step=step,
                    seconds=time.time() - start,
                    metrics=result.metrics,
                )
                opt = optimizer.result(timeout=1800)
                event(
                    "train_done",
                    step=step,
                    seconds=time.time() - start,
                    metrics=opt.metrics,
                )
            event("complete")
        except BaseException as exc:
            barrier.abort()
            event("failed", error=f"{type(exc).__name__}: {exc}")
            raise

    with ThreadPoolExecutor(args.clients) as pool:
        list(pool.map(client, range(args.clients)))
    (args.output / "COMPLETE").write_text("all clients completed\n")


if __name__ == "__main__":
    main()
