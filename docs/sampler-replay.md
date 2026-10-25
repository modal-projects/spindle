# Sampler replay

Spindle can train using token sampling supports captured by SGLang. The Miles
LoRA backend also supports expert-route replay. Capture is opt-in per request:

- `return_routed_experts=True` captures the selected MoE expert IDs. Training
  reuses those selections while computing routing weights from its own logits.
- `return_sampling_mask=True` captures the allowed token IDs after top-k,
  top-p, and min-p filtering, together with log probabilities normalized over
  that support. Training normalizes over the same support and temperature.
- `return_sampling_probabilities=True` additionally captures normalized log
  probabilities for **every** candidate in that support, for exact score centering.
  This implies `return_sampling_mask=True`.

Ordinary Tinker sampling discards additional response fields. Use
`spindle.sample_with_replay()` to receive them, then pass its replay tensors through
`Datum.loss_fn_inputs`. Megatron supports token replay with the native `mis`,
`score_centered`, and `score_centered_mis` losses. Router replay remains specific
to the Miles LoRA backend.

## Native MIS and score centering

For full-parameter Megatron training, request `return_sampling_probabilities=True`
and call `training.forward_backward(data, "score_centered_mis", loss_fn_config)`.
Include `**sequence.replay.loss_fn_config()` in the config, plus `mis_low` and
`mis_high` (defaults 0.2 and 5.0). No custom-forward/backward callback is needed.

For each target, let `q` be the recorded sampler probabilities and `p` the
trainer's probabilities renormalized on exactly the recorded support. The
native loss has gradient

```text
-A * (w[y] * grad(log p[y]) - sum_v q[v] * w[v] * grad(log p[v]))
w[v] = (p[v] / q[v]) * indicator(mis_low <= p[v] / q[v] <= mis_high)
```

Both importance weights and correction coefficients are detached. The correction
remains when the sampled action is masked. `score_centered` uses `w=1`; `mis`
omits the correction. Candidate probabilities are required for both centered
losses; a chosen-token log probability alone cannot determine the correction.

The bundled SGLang patch supports exact candidate probabilities with
`sglang_cfg = {"sampling_backend": "pytorch", ...}`, without speculative decoding
or prefill/decode disaggregation. It rejects unsupported modes. Set `top_k=-1`
explicitly when intending pure top-p sampling. A support that exceeds
`sampling_mask_max_tokens` is an error, never silently truncated.

`sampling_mask_logprobs` is a float32 tensor aligned one-to-one with the flattened
`sampling_mask_ids`; offsets have one entry per target plus a final endpoint.
Native replay gathers only support logits across tensor-parallel ranks and
preserves the original positions across context partitioning and packed padding.
Metrics include entropy on the replayed support, `KL(q || p)` on that support,
and the fraction of sampled tokens rejected by MIS.

See [the mixed-sampler reproduction](../examples/mixed-samplers/README.md) for a
token-preserving, multi-turn asynchronous training loop.

## Server configuration

Sampling-mask capture needs no additional server flag. For router capture,
use a MoE model and set `ROLLOUT_RETURN_ROUTED_EXPERTS = True` in its rollout
pool definition. This adds SGLang's `--enable-return-routed-experts` flag.
Also append `--use-rollout-routing-replay` to the trainer's
`backend_config["miles"]["extra_args"]` before deployment.

Router capture increases the rollout server's memory use, and captured supports
can make responses large when top-p retains much of the vocabulary. Neither
capture is requested by default.

## Sample and train

This example assumes `training` is an existing Miles LoRA training client.
It uses a constant advantage to demonstrate the handoff; replace that with
advantages computed from your task's rewards.

```python
import spindle
from tinker import types

sampler = training.save_weights_and_get_sampling_client(name="replay-step")
tokenizer = training.get_tokenizer()
prompt_tokens = tokenizer.encode("The capital of France is", add_special_tokens=True)
prompt = types.ModelInput.from_ints(prompt_tokens)
sequence = spindle.sample_with_replay(
    sampler,
    prompt,
    sampling_params=types.SamplingParams(max_tokens=32, temperature=0.8, top_p=0.9),
    return_sampling_mask=True,
).result().sequences[0]

if not sequence.tokens:
    raise RuntimeError("No generated tokens to train on")
tokens = prompt_tokens + sequence.tokens
replay_inputs = sequence.replay.training_inputs(len(sequence.tokens))
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": tokens[1:],
        "advantages": [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(sequence.tokens),
        **replay_inputs,
    },
)
forward = training.forward_backward(
    [datum],
    loss_fn="ppo",
    loss_fn_config=sequence.replay.loss_fn_config(),
)
optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))
print(forward.result().metrics)
print(optimizer.result().metrics)
```

For MoE router replay, additionally request `return_routed_experts=True` and
supply the model's **total transformer layer count** and routed experts per token:

```python
replay_inputs = sequence.replay.training_inputs(
    len(sequence.tokens),
    num_layers=model_config.num_hidden_layers,
    experts_per_token=model_config.num_experts_per_tok,
)
```

SGLang returns an unshaped base64 int32 routing buffer, so these dimensions must
match the deployed model. Every datum in a router-replayed request must provide
routes. The scheduler separates these requests from ordinary requests.

## Alignment and probability semantics

`training_inputs()` assumes one sampled continuation and an input of
`(prompt_tokens + sequence.tokens)[:-1]`. Expert routes cover those input
positions; sampling supports cover the generated targets. The helper inserts
empty, unrestricted support rows for prompt targets. Set prompt advantages or
cross-entropy weights to zero.

The helper supplies `logprobs` from the **masked sampling distribution**.
Spread its result after any other loss inputs so ordinary model log probabilities
do not overwrite them. `sequence.logprobs` retains the existing API meaning.
Always pass `replay.loss_fn_config()` to preserve the sampling temperature.

The mask is frozen from rollout: training does not recompute top-p from its
updated logits. Replay does not guarantee bitwise parity between SGLang and
Megatron kernels. For multi-turn trajectories or truncated sequences, callers
must realign replay tensors to their actual training inputs; the convenience
helper does not stitch turns.
