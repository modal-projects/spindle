"""Sample from whichever of two pools is up first, then train one step.

Deploy `groups.TRAINER` for the training client. For each group, build a rollout
app with `FFTPoolSpec(deployment.definition_id, training.model_id, True, 0)` so
both pools read that trainer's bulletin. Pass the server URLs here. The script
does not wait for the slower pool after the update finishes.
"""

from __future__ import annotations

import argparse
import asyncio
import os

import httpx
import tinker
from tinker import types

import spindle
from readiness import SamplerGate
from spindle.client import create_full_training_client

from groups import GROUPS

TIMEOUT = 60 * 60


async def probe(url):
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                response = await client.get(url)
            except httpx.HTTPError:
                await asyncio.sleep(2)
                continue
            if response.status_code < 500:
                return url
            await asyncio.sleep(2)


async def run(urls):
    missing = [group["name"] for group in GROUPS if group["name"] not in urls]
    if missing:
        raise SystemExit(f"pass a URL for {', '.join(missing)}")
    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
    )
    training = create_full_training_client(service, "Qwen/Qwen3.5-4B")
    training.save_weights_for_sampler("initial").result(timeout=TIMEOUT)
    gate = SamplerGate()
    await gate.open({name: probe(url) for name, url in urls.items()})
    print(f"sampling with {sorted(gate.ready)}")
    adopting = asyncio.create_task(gate.adopt())
    try:
        sampling = training.save_weights_and_get_sampling_client()
        tokenizer = training.get_tokenizer()
        prompt = tokenizer.encode("What is 2 + 2?", add_special_tokens=True)
        sequence = (
            spindle.sample_with_replay(
                sampling,
                types.ModelInput.from_ints(prompt),
                sampling_params=types.SamplingParams(
                    max_tokens=8, temperature=1.0, top_p=0.95
                ),
                return_sampling_probabilities=True,
            )
            .result(timeout=TIMEOUT)
            .sequences[0]
        )
        tokens = prompt + list(sequence.tokens)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "advantages": [0.0] * (len(prompt) - 1) + [1.0] * len(sequence.tokens),
                **sequence.replay.training_inputs(len(sequence.tokens)),
            },
        )
        result = training.forward_backward(
            [datum],
            "score_centered_mis",
            loss_fn_config=sequence.replay.loss_fn_config(),
        ).result(timeout=TIMEOUT)
        training.optim_step(types.AdamParams(learning_rate=1e-5)).result(
            timeout=TIMEOUT
        )
        print(result.metrics)
    finally:
        adopting.cancel()
        await asyncio.gather(adopting, return_exceptions=True)
        print(
            f"ready={sorted(gate.ready)} pending_cancelled={not gate.pending_names()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", required=True, metavar="NAME=URL")
    args = parser.parse_args()
    urls = {}
    for item in args.url:
        name, _, url = item.partition("=")
        if not url:
            raise SystemExit(f"expected NAME=URL, got {item}")
        urls[name] = url
    asyncio.run(run(urls))


if __name__ == "__main__":
    main()
