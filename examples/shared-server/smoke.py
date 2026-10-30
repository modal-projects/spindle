"""Train two LoRA steps on a deployed Spindle server.

Reads TINKER_BASE_URL and TINKER_API_KEY, plus MODAL_PROXY_TOKEN_ID and
MODAL_PROXY_TOKEN_SECRET when the server requires Modal proxy auth. With
--expect-commit, waits until the server reports that commit before training,
so a deploy job can confirm its rollout and that the recipe still trains.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import tinker
from tinker import types
from with_proxy_auth import install, proxy_headers


def wait_for_commit(base_url, api_key, commit, timeout_s):
    url = (
        base_url.rstrip("/").removesuffix("/api/v1") + "/api/v1/get_server_capabilities"
    )
    deadline = time.monotonic() + timeout_s
    while True:
        request = urllib.request.Request(
            url, headers={"x-api-key": api_key, **proxy_headers()}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                served = json.load(response).get("spindle_commit") or ""
        except OSError as error:
            served = f"unreachable ({error})"
        if served.startswith(commit):
            print(f"server is at {served}")
            return
        if time.monotonic() > deadline:
            sys.exit(f"server is at {served}, expected {commit}")
        time.sleep(15)


def train(base_url, api_key, base_model, steps):
    service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
    training = service.create_lora_training_client(base_model=base_model, rank=8)
    tokens = training.get_tokenizer().encode("The shared server trains one step.")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )
    for step in range(steps):
        started = time.monotonic()
        result = training.forward_backward([datum], "cross_entropy").result()
        training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        loss = result.metrics["loss:sum"] / (len(tokens) - 1)
        if not math.isfinite(loss):
            sys.exit(f"step {step}: loss {loss}")
        print(f"step {step}: loss {loss:.3f} in {time.monotonic() - started:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--expect-commit")
    parser.add_argument(
        "--commit-timeout",
        type=int,
        default=600,
        help="seconds to wait for --expect-commit",
    )
    args = parser.parse_args()
    base_url = os.environ["TINKER_BASE_URL"]
    api_key = os.environ["TINKER_API_KEY"]
    if proxy_headers():
        install()
    if args.expect_commit:
        wait_for_commit(base_url, api_key, args.expect_commit, args.commit_timeout)
    train(base_url, api_key, args.base_model, args.steps)


if __name__ == "__main__":
    main()
