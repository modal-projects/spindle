# Score-centering loss

Use `loss_fn="score_centering"` to opt into the top-k correction from
[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807).
Both the Miles LoRA backend and the Megatron LoRA/full-training backends support
the loss. Existing losses and sampling defaults are unchanged.

For each target position, the loss is

```text
rho = max(1 - sum(q_head), epsilon) / max(1 - sum(p_head), epsilon)
L = -advantage * (log p_sampled - sum(stop_gradient(q_head - rho*p_head) * log p_head))
```

Here `p` is the trainer softmax and `q` is the rollout sampler distribution.
The logged head probabilities retain their full-vocabulary normalization.
The unlogged tail is approximated by the trainer distribution scaled to match
the sampler's remaining mass. This is a correction to the policy gradient;
compute rewards/advantages as usual before submitting training data.

## Sample and train

Request top-k capture only for rollouts that will use this loss. The helper
returns the usual sampled tokens plus a `.replay` object containing top-k IDs
and probabilities at each generated token.

```python
import spindle
from tinker import types

# training is an existing Spindle training client.
sampler = training.save_weights_and_get_sampling_client(name="score-centering")
prompt_tokens = training.get_tokenizer().encode("Solve 3x + 4 = 19.")
sequence = spindle.sample_with_replay(
    sampler,
    types.ModelInput.from_ints(prompt_tokens),
    sampling_params=types.SamplingParams(
        max_tokens=1024, temperature=1.0, top_p=1.0, top_k=-1,
    ),
    score_centering_topk=128,
).result().sequences[0]

if not sequence.tokens:
    raise RuntimeError("No generated tokens to train on")

tokens = prompt_tokens + sequence.tokens
advantage = 1.0  # Replace with your reward-derived, client-normalized advantage.
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": tokens[1:],
        "advantages": [0.0] * (len(prompt_tokens) - 1)
        + [advantage] * len(sequence.tokens),
        **sequence.replay.training_inputs(len(sequence.tokens)),
    },
)
result = training.forward_backward(
    [datum], loss_fn="score_centering",
    # Optional composition with truncated importance sampling:
    # loss_fn_config={"importance_sampling_clip": 2.0},
).result()
training.optim_step(types.AdamParams(learning_rate=1e-5)).result()
```

The initial capture path requires **temperature 1, top_p 1, and top_k -1**.
SGLang's ordinary top-logprobs are measured before sampling truncation; this
restriction ensures they describe the distribution that generated the tokens.
Requests with incompatible settings fail instead of silently using the wrong
distribution. Sampling-mask replay cannot be combined with this loss. Miles
MoE expert-route replay can be requested separately with
`return_routed_experts=True`, using the dimensions and server configuration
documented in [sampler replay](sampler-replay.md).

Capture widths from 1 to 128 are accepted. The paper uses 128 and reports
ablations at 32. Capture adds O(generated tokens × k) response/storage overhead;
ordinary sampling does not request these tensors. Existing rollouts containing
only sampled-token logprobs cannot be retrofitted with this correction.

## Training inputs and configuration

Alongside the usual `target_tokens`, `logprobs`, and `advantages`, supply:

| Input | Tensor shape and type | Meaning |
| --- | --- | --- |
| `sampling_topk_token_ids` | `[input_tokens, k]`, int64 | Sampler head IDs at each target position; -1 marks padding. |
| `sampling_topk_logprobs` | `[input_tokens, k]`, float32 | Corresponding full-distribution log probabilities; use zero for padding. |

Set prompt/tool/padding advantages to zero. Every position with a nonzero
advantage needs captured head probabilities. The helper aligns one prompt and
one continuation with `(prompt + response)[:-1]`; for multi-turn trajectories,
align each turn's tensors to the actual training sequence, inserting -1/zero
head rows for intervening observations. Do not recompute the head from a later
sampler version.

Loss options are per `forward_backward` call:

| Option | Default | Meaning |
| --- | --- | --- |
| `tail_mass_epsilon` | `1e-6` | Positive floor for both tail masses, as in the paper. Must be less than one. |
| `importance_sampling_clip` | absent | Enable truncated importance weights `min(p/q, clip)`; positive finite clip required. |

With importance weighting, the correction centers the **weighted** score:
the head coefficient becomes `q * min(p/q, clip) - alpha * p`, where
`alpha = min(1, rho * clip)`. Both this coefficient and the sampled importance
weight are detached (paper equation 14). This is a standalone loss, not an
additive penalty on PPO.

Normalization follows the existing backend conventions. Miles sums the
client-supplied advantages without extra normalization; include any desired
token/batch scaling in those advantages and omit `weights`. Megatron retains
its existing loss mask and training normalization. No optimizer defaults change.

## Validation

CPU tests compare gradients against a full-vocabulary reference, check zero
expected drift under constant rewards, and cover on-policy equivalence, clipped
importance weights, saturated probabilities, prompt/padding masks, packing
different head widths, and two-rank tensor/context parallelism. SDK tests cover
capture, JSON/protobuf transport, and training submission. A GPU training run
is still needed to measure runtime overhead and learning behavior.
