# GLM-5.3 Flash LoRA (experimental)

This recipe connects `zai-org/GLM-5.3-Flash` to Spindle's Miles trainer and shared
SGLang sampling pool. It supports text inputs with a 16,384-token context, including
the prompt and generated response.

```bash
spindle config init --preset glm53-flash-lora-16k > glm53.py
spindle config validate glm53.py
spindle deploy glm53.py
```

The proposed allocation is two trainer nodes with eight H200s each, plus eight
H200s per sampling replica. The trainer uses TP8, EP16 and CP1. Up to four clients
share it, with adapter ranks up to 32. The sampler initially allows one replica;
raise `inference_max_replicas` to add serving capacity. Full-model memory use and
multi-node execution are still being validated.

Use an ordinary Tinker LoRA client with `base_model="zai-org/GLM-5.3-Flash"` and
`train_unembed=False`. Publishing an adapter loads it into the same model's sampling
pool. Checkpoints use Miles' per-slot model and optimizer state.

The recipe trains KDA Q/K/V/output projections, DSA query/down projections and
output projections, and dense, shared and routed MLP experts. The DSA `kv_b_proj`
adapter is excluded: the upstream absorbed-attention implementation handles one
adapter delta, which does not support batches containing different clients.
The KDA gates, indexer, hyper-connections and vision encoder stay frozen.
Context parallelism is currently unsupported by the upstream KDA provider.

The public checkpoint stores FP8 weights. The trainer imports them as BF16 and
keeps the base weights frozen. SGLang serves the FP8 base with BF16 KV cache and
TileLang sparse attention on H200s.

This recipe selects its own `trainer_image` and `inference_image`, written as
`module:attribute` references to Modal images. Changing either reference redeploys
the corresponding app. The GLM images retain Spindle's Miles revision, update
Bridge for Transformers 5.16 support, and add the model packages from
[Megatron-Bridge PR 35](https://github.com/radixark/Megatron-Bridge/pull/35), which
is still under review. Exact dependency revisions are in
[glm53_image.py](../../src/spindle/providers/modal/glm53_image.py).

Run the image import and configuration checks with:

```bash
PYTHONPATH=src modal run tests/manual/validate_glm53.py
```

Add `--train` to run two optimizer updates with two adapter slots on the published
48.7 GB four-layer checkpoint. This checks checkpoint restore and PEFT export on
one H200. Checkpoint restore compares all exported adapter tensors exactly;
repeated training forwards can produce different logprobs even with unchanged
weights. Add `--sample` to load the exported adapter in a normal SGLang server
and compare its logprobs with training. The weights are cached in the
`spindle-glm53-pr26-validation` volume. Running `--sample` alone reuses the latest
export without retraining.
These checks do not load the full checkpoint or demonstrate training convergence.
