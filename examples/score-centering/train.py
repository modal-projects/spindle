from __future__ import annotations

import os

import tinker
from tinker import types

import spindle
from spindle.client import create_full_training_client

BASE_MODEL = "Qwen/Qwen3.5-4B"
TIMEOUT = 60 * 60


def main() -> None:
    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
    )
    training = create_full_training_client(service, BASE_MODEL)
    tokenizer = training.get_tokenizer()
    prompt = tokenizer.encode(
        "What is 2 + 2? Answer with only the number.",
        add_special_tokens=True,
    )
    sampling = training.save_weights_and_get_sampling_client()
    sequence = (
        spindle.sample_with_replay(
            sampling,
            types.ModelInput.from_ints(prompt),
            sampling_params=types.SamplingParams(
                max_tokens=16, temperature=1.0, top_p=0.95
            ),
            return_sampling_probabilities=True,
        )
        .result(timeout=TIMEOUT)
        .sequences[0]
    )
    if not sequence.tokens:
        raise RuntimeError("sampling returned no tokens")
    response_text = tokenizer.decode(sequence.tokens)
    reward = 1.0 if "4" in response_text else -1.0
    tokens = prompt + list(sequence.tokens)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "advantages": [0.0] * (len(prompt) - 1) + [reward] * len(sequence.tokens),
            **sequence.replay.training_inputs(len(sequence.tokens)),
        },
    )
    forward = training.forward_backward(
        [datum],
        "score_centered_mis",
        loss_fn_config=sequence.replay.loss_fn_config(),
    )
    optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))
    print(f"response: {response_text!r}")
    print(f"reward: {reward}")
    print(f"training metrics: {forward.result(timeout=TIMEOUT).metrics}")
    print(f"optimizer metrics: {optimizer.result(timeout=TIMEOUT).metrics}")


if __name__ == "__main__":
    main()
