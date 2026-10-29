"""Bounded two-client native-loss/optimizer/publication validation, not RL evidence."""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx
import modal
import tinker
from tinker import types

from .miles_config import config
from .publication import latest_ref
from .train import tensor


def validation_datum(sign):
    tokens = [100] * 64 + [0, 2, 3] * 21 + [2]
    targets = tokens[1:]
    active = [int(i >= 63) for i in range(len(targets))]
    supports, offsets, logq = [], [0], []
    probabilities = {0: 0.1, 2: 0.3, 3: 0.6}
    for token, trained in zip(targets, active, strict=True):
        if trained:
            supports.extend([0, 2, 3])
            logq.extend(math.log(q) for q in probabilities.values())
        offsets.append(len(supports))
    return types.Datum(
        types.ModelInput.from_ints(tokens[:-1]),
        {
            "target_tokens": tensor(targets, "int64"),
            "weights": tensor([a / sum(active) for a in active]),
            "advantages": tensor([sign * a for a in active]),
            "logprobs": tensor(
                [
                    math.log(probabilities[t]) if a else 0.0
                    for t, a in zip(targets, active, strict=True)
                ]
            ),
            "sampling_mask_ids": tensor(supports, "int64"),
            "sampling_mask_offsets": tensor(offsets, "int64"),
            "sampling_mask_logprobs": tensor(logq),
        },
    )


async def validate(args, service, url, key):
    events = []
    clients = []
    args.output.mkdir(parents=True, exist_ok=True)

    def event(kind, **fields):
        row = {"event": kind, "at": time.time(), **fields}
        events.append(row)
        with (args.output / "events.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    loss_config = {"sampling_temperature": 1.0, "mis_low": 0.2, "mis_high": 5.0}

    async def result(future):
        return await asyncio.wait_for(future.result_async(), 1800)

    async def fingerprint(client, name):
        publication = await result(client.save_weights_for_sampler(name))
        volume = modal.Volume.from_name(
            config.platform["storage"]["bulletin"], environment_name="kailash-dev"
        )
        ref = await latest_ref(client.model_id, config.platform["storage"]["bulletin"])
        path = ref.identity + "/snapshot.json"
        manifest = json.loads(
            await asyncio.to_thread(lambda: b"".join(volume.read_file(path)))
        )
        digest = manifest["files"]["adapter_model.safetensors"]["sha256"]
        event(
            "adapter_fingerprint",
            model_id=client.model_id,
            publication=publication.path,
            version=ref.version,
            sha256=digest,
        )
        return digest

    try:
        for index in range(2):
            event("creating_client", client=index)
            client = await asyncio.wait_for(
                service.create_lora_training_client_async(
                    config.model,
                    rank=32,
                    train_attn=True,
                    train_mlp=True,
                    train_unembed=False,
                ),
                3600,
            )
            clients.append(client)
            event("client_created", client=index, model_id=client.model_id)
        before_hashes = [await fingerprint(c, "isolation-before") for c in clients]
        baselines = [
            (
                await result(
                    clients[1].forward(
                        [validation_datum(1.0)], "score_centered_mis", loss_config
                    )
                )
            )
            .loss_fn_outputs[0]["logprobs"]
            .data
            for _ in range(3)
        ]
        first = await result(
            clients[0].forward_backward(
                [validation_datum(1.0)], "score_centered_mis", loss_config
            )
        )
        update = await result(
            clients[0].optim_step(
                types.AdamParams(learning_rate=1e-5, grad_clip_norm=1.0)
            )
        )
        untouched_after = await result(
            clients[1].forward(
                [validation_datum(1.0)], "score_centered_mis", loss_config
            )
        )
        after_hashes = [await fingerprint(c, "isolation-after") for c in clients]
        after = untouched_after.loss_fn_outputs[0]["logprobs"].data
        baseline_delta = max(
            abs(a - b)
            for row in baselines[1:]
            for a, b in zip(baselines[0], row, strict=True)
        )
        after_delta = max(abs(a - b) for a, b in zip(baselines[0], after, strict=True))
        event(
            "isolation_measurements",
            baseline_logprob_max_delta=baseline_delta,
            after_logprob_max_delta=after_delta,
            before_hashes=before_hashes,
            after_hashes=after_hashes,
            backward=first.metrics,
            optimizer=update.metrics,
        )
        if before_hashes[1] != after_hashes[1]:
            raise RuntimeError("updating client 0 changed client 1's adapter weights")
        if before_hashes[0] == after_hashes[0] or update.metrics["grad_norm:mean"] <= 0:
            raise RuntimeError("client 0 did not make a nonzero adapter update")
        if after_delta > max(1e-5, 3 * baseline_delta):
            raise RuntimeError(
                "untouched client logits changed beyond observed numerical variation"
            )
        event(
            "adapter_isolation",
            exact=True,
            baseline_logprob_max_delta=baseline_delta,
            after_logprob_max_delta=after_delta,
            backward=first.metrics,
            optimizer=update.metrics,
        )

        async def loop(index, client):
            await asyncio.sleep(index * args.stagger_seconds)
            for step in range(2):
                event("step_started", client=index, step=step)
                result = await asyncio.wait_for(
                    client.forward_backward(
                        [validation_datum(1.0 if index == 0 else -1.0)],
                        "score_centered_mis",
                        loss_config,
                    ).result_async(),
                    1800,
                )
                event(
                    "backward_completed",
                    client=index,
                    step=step,
                    metrics=result.metrics,
                )
                optimizer = await asyncio.wait_for(
                    client.optim_step(
                        types.AdamParams(
                            learning_rate=1e-6,
                            beta1=0.9,
                            beta2=0.95,
                            eps=1e-8,
                            grad_clip_norm=1.0,
                        )
                    ).result_async(),
                    1800,
                )
                if optimizer.metrics["update_successful:mean"] != 1 or not all(
                    math.isfinite(v)
                    for v in {**result.metrics, **optimizer.metrics}.values()
                ):
                    raise RuntimeError("non-finite or skipped Miles update")
                publication = await asyncio.wait_for(
                    client.save_weights_for_sampler(
                        f"validation-{step}"
                    ).result_async(),
                    1800,
                )
                event(
                    "step_completed",
                    client=index,
                    step=step,
                    publication=publication.path,
                    optimizer=optimizer.metrics,
                )
            checkpoint = await client.save_state("validated").result_async()
            event("checkpoint", client=index, path=checkpoint.path)

        await asyncio.gather(*(loop(i, client) for i, client in enumerate(clients)))
        starts = {
            (e["client"], e["step"]): e["at"]
            for e in events
            if e["event"] == "step_started"
        }
        ends = {
            (e["client"], e["step"]): e["at"]
            for e in events
            if e["event"] == "step_completed"
        }
        independent = any(
            starts[other, 0] < starts[client, 1] < ends[other, 0]
            for client, other in ((0, 1), (1, 0))
        )
        if not independent:
            raise RuntimeError(
                "validation did not observe a client advancing independently"
            )
        (args.output / "result.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "backend": "miles",
                    "parameterization": "lora",
                    "model_ids": [c.model_id for c in clients],
                    "adapter_isolation_exact": True,
                    "independent_client_progress": independent,
                    "interpretation": "Synthetic integration validation; not a training-stability result.",
                },
                indent=2,
            )
        )
        event("validation_passed")
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
    parser.add_argument("--stagger-seconds", type=float, default=60.0)
    args = parser.parse_args()
    url = modal.Function.from_name(
        config.platform["frontend"], "server", environment_name="kailash-dev"
    ).get_web_url()
    key = args.api_key_file.read_text().strip()
    service = tinker.ServiceClient(base_url=url, api_key=key)
    asyncio.run(validate(args, service, url, key))


if __name__ == "__main__":
    main()
