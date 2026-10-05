# GLM-5.3 Flash LoRA (experimental)

This recipe connects `zai-org/GLM-5.3-Flash` to Spindle's Miles trainer and shared
SGLang sampling pool. It configures text inputs with a 16,384-token context, including
the prompt and generated response.

```bash
spindle config init --preset glm53-flash-lora-16k > glm53.py
spindle config validate glm53.py
spindle deploy glm53.py
```

The allocation is four trainer nodes with eight H200s each, plus eight
H200s per sampling replica. The trainer uses TP8, EP32 and CP1. Up to four clients
share it, with adapter ranks up to 32. The sampler initially allows one replica;
raise `inference_max_replicas` to add serving capacity.

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
TileLang sparse attention on H200s. Shared experts stay separate from routed
experts so their adapters remain addressable. The sampler image also fixes LoRA
buffer dimensions for the model's alternating KDA and DSA layers.

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
The four-layer test passed two updates, checkpoint restore, adapter export and
eight-token generation through SGLang. The mean trainer/sampler logprob difference
was 0.049 over 32 scored tokens. Full-model FP8 serving remains unvalidated.

For the full model on four trainer nodes, run:

```bash
PYTHONPATH=src modal run --detach tests/manual/glm53_multinode.py
```

This downloads the checkpoint on CPU before allocating 32 H200s. It runs two
256-token updates and one 16K update across four rank-32 adapters, checks
checkpoint restore, and records adapter export and publication times. The test
uses Spindle's Ray cluster startup and Miles runtime; it does not exercise the
HTTP API or launch inference. The cluster stops when the test finishes.
Artifacts are stored in `spindle-glm53-pr26-full-validation`.

On October 5, the full 45-layer model passed all three updates and exact adapter
comparison after checkpoint restore on 32 H200s (TP8/EP32):

| Operation | Wall time |
| --- | ---: |
| Miles runtime startup, checkpoint already cached | 466.4 s |
| First update: eight 256-token sequences | 371.7 s |
| Second update: eight 256-token sequences | 4.84 s |
| First 16K update: four 16,384-token sequences | 188.4 s |
| Optimizer after the second and third updates | 0.16–0.17 s |
| PEFT export, before publication changes | 92.9 s |
| Publication: hash, copy and commit | 29.3 s |
| Verification of the published file on the same worker | 15.5 s |
| Save / restore training checkpoint | 70.3 / 29.2 s |

The first short update and the first 16K update included kernel compilation.
The 16K result is not a steady-state throughput measurement. These are Miles
runtime call times with synthetic token batches; SDK and HTTP latency are excluded.
The export used the validation checkpoint volume, including its synchronization
and commit path. Publication now stages on the bulletin volume and skips the
unrelated checkpoint-volume commit.


With the configured targets, one rank-32 adapter contains approximately 7.23B
exported tensor elements, or 14.47 GB in BF16 (14.48 GB including the file header). Routed expert MLPs account for 98.6% of this.
Each published version writes the complete adapter. Miles stages the export on
the bulletin volume, then publication hashes and atomically renames it into place.
The sampler verifies the contents before loading a new version.

The GLM image loads each worker's local experts into CPU memory before normalizing
the adapter. Preparation runs on a background thread while the scheduler continues
serving. All TP ranks must finish successfully before registration completes.
GPU buffer placement still happens through SGLang's existing LoRA memory pool.
The sampler keeps up to eight registered versions in CPU memory and four on GPU,
with 512 GiB of host memory requested. The BF16 test below measures the full-model
adapter cache.

Run the installed-loader checks and TP2/EP2 serving measurement with:

```bash
PYTHONPATH=src modal run --detach tests/manual/validate_glm53_loading.py
```

This checks expert ownership, failed and cancelled preparation, and TP readiness.
The serving measurement loads a second adapter while an existing adapter generates
128 tokens, recording registration time and streaming progress during the load.

The TP2/EP2 four-layer test retained 132.2 MB per worker from a 260.3 MB adapter.
Registration during an active decode took 2.27 s, with 10 streaming chunks received
before it completed. The largest chunk gap was 0.86 s. This verifies that CPU
preparation can overlap serving. Full-model BF16 measurements are below.

To compare publication methods on the full exported adapter without allocating GPUs:

```bash
PYTHONPATH=src modal run --detach tests/manual/glm53_publication.py
```

This alternates copy and rename publication, checks the resulting files, and records
hashing and commit time together. Staging is measured separately. It does not
measure GPU adapter capture or cold sampler downloads.

On the 14.48 GB adapter, copy publication took 32.5 and 32.8 s; rename publication
took 16.3 and 15.4 s. The order was copy, rename, rename, copy. Mean publication
time fell from 32.7 to 15.8 s (52%). Hashing remains, and verification added another
13.9–15.7 s. These timings use staged files on the validation volume; they exclude
GPU export, staging, and cold reads on a separate sampling replica.

## BF16 RL integration test

These commands reuse the model and exported adapter from the four-node test above.
To test full-model rollouts without FP8 serving, first cache a BF16 checkpoint:

```bash
PYTHONPATH=src modal run --detach tests/manual/prepare_glm53_bf16.py
PYTHONPATH=src modal run --detach tests/manual/validate_glm53_rl.py
```

Conversion uses the same blockwise FP8-to-BF16 routine as the Miles trainer and
stores the result in the full-validation volume. It resumes completed shards and
uses up to four preparation GPUs. The BF16 checkpoint occupies 642.65 GB. Conversion
only needs to run once for the cached source checkpoint.

The integration test starts an eight-GPU BF16 sampler (H200, with B200 as a
capacity fallback) and checks the previously exported full adapter before
allocating the four training nodes. It runs two
updates with one active rank-32 adapter, two math problems per update, four
responses per problem, and a 1,024-token generation cap. Context capacity remains
16K. Rewards combine exact boxed-answer correctness with a small length penalty;
group-normalized advantages feed the PPO objective with ratio bounds 0.8 and 1.2
and learning rate 1e-5.

The loop uses Spindle's Miles command backend, snapshot publication, and LoRA
sampling sidecar. It checks finite nonzero gradients, the served adapter version,
and a changed fixed-prompt logprob after each update. The SDK frontend and
EngineServer scheduling are outside this test. Reports and sampled responses are
saved in `spindle-glm53-pr26-rl-checkpoints`.

The BF16 test limits checkpoint loading to two threads per worker to bound CPU
staging memory. SGLang's default eight-thread loader temporarily retained roughly
1 TB across the eight workers during the first full-model load. This is separate
from the per-adapter CPU cache.

The first full-model BF16 serving check loaded the 14.48 GB adapter in 96.4 s
and generated tokens successfully. Each TP8/EP8 worker retained 1,983,807,488 bytes
(1.98 GB) of adapter tensors on CPU. This counts tensor storage, excluding Python
objects and temporary loading buffers. Registration used a cold adapter; it
does not measure repeated GPU cache swaps or loading during an active decode.
