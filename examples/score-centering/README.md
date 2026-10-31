# Score centering

One full-parameter client samples with top-p, records a normalized log
probability for every token in the sampler's support, and trains with
`score_centered_mis`.

The sampler still publishes BF16 weights. `return_sampling_probabilities=True`
asks SGLang for `sampling_logprobs_mode="support"`. A support larger than the
server's `sampling_mask_max_tokens` is an error. `mis` drops the centering
term, and `score_centered` uses an importance weight of one. Ratio bounds are
`mis_low` and `mis_high` in `loss_fn_config` (0.2 and 5 when omitted).

Deploy a Qwen3.5-4B full-parameter recipe, then from a project that depends on
this checkout:

```bash
TINKER_BASE_URL=... TINKER_API_KEY=... uv run python examples/score-centering/train.py
```

Miles LoRA accepts the same loss names. See [sampler replay](../../docs/sampler-replay.md).
