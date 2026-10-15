# Design: quantized training and inference

Status: proposal. This PR changes documentation only. Configuration examples below
are proposed API, not currently supported recipes. No FP8 performance or training
quality result is claimed.

## Goal and scope

Allow a recipe to select FP8 inference, optionally use FP8 training compute, and
publish weights in the representation the inference engine expects. Keep the
sampling/training API, independent client loops, and publication lifecycle intact.

Packing at export is the right boundary for changing base weights. It is not the
whole change: startup assets, scale metadata, version-zero state, and the receiver
must agree on the same representation. Training precision is a separate choice.

Three cases must remain distinct:

| Case | Trainer | Publication | Inference |
| --- | --- | --- | --- |
| LoRA, FP8 serving | BF16 base and trainable adapters initially | BF16 adapters; prepare frozen FP8 base once | FP8 base plus BF16 adapters |
| LoRA, FP8 training compute | Eligible base operations use backend FP8 kernels; adapters and optimizer retain their configured precision | Same adapter-only publication | Same FP8 base plus BF16 adapters |
| Full-weight training, FP8 serving | BF16 or separately enabled FP8 compute | Pack every published base snapshot, including scales | Load a complete FP8 version |

FP8 compute does not imply persistent FP8 parameter storage, FP8 optimizer state,
quantized adapters, or FP8 KV cache. Those are different features. Start with
E4M3 weights, 128-by-128 blocks, FP32 dequantization scales, and dynamic activation
scaling on a validated Hopper backend. Do not silently substitute MXFP8, NVFP4,
power-of-two scales, or a different block layout on other hardware.

## Existing boundaries

Source inspected: Spindle `c73680e`, `modal-projects/miles:spindle-miles`
`9d047db`, and `modal-projects/sglang:spindle-sglang-v0.5.21` `b90efa27`.
These are source references, not proof that a deployed image contains them.

- [BaseConfig](../src/spindle/configuration.py) already exposes flat recipe fields,
  backend dictionaries, and dotted overrides. [Backend resolution](../src/spindle/backends/deployment.py)
  currently allows Miles for LoRA and Megatron for full training. Inference options
  are resolved separately, but there is no shared quantization contract.
- [Miles LoRA capture](../src/spindle/backends/miles_lora.py) calls
  `export_slot_peft`, then publishes the detached adapter artifact through the
  existing bulletin. It does not republish the frozen base at each update.
- [Full-weight capture](../src/spindle/backends/megatron_runtime/fft/delta.py)
  exports HF weights through Megatron Bridge and computes byte deltas against a
  previous checkpoint. Its initial tensor set comes from that checkpoint; capture
  indexes by name and casts to the previous tensor's dtype. Adding scales or
  changing BF16 to FP8 is therefore not a transparent substitution.
- [The full-weight sidecar](../src/spindle/inference/full_sidecar.py) uses Stitch's
  SGLang engine with CPU delta staging. SGLang's
  [rank compiler](https://github.com/modal-projects/sglang/blob/b90efa27081313d9cc6d295470ad4ef8d3b41d55/python/sglang/srt/weight_sync/rank_weight_compiler.py)
  already loads canonical checkpoints into isolated rank-local weight images and
  invokes quantization postprocessing. Some postprocessing requires GPU work;
  CPU staging does not mean all preparation is CPU-only.

### What to reuse from Miles

Miles already has a [streaming HF weight iterator](https://github.com/modal-projects/miles/blob/9d047db23a5760414de12185b3a133363ca2fde8/miles/backends/training_utils/weight_update/hf_weight_iterator/__init__.py)
with bounded buckets and atomic update groups. Its
[Bridge conversion](https://github.com/modal-projects/miles/blob/9d047db23a5760414de12185b3a133363ca2fde8/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py)
applies quantization to base tensors, not LoRA tensors, and groups a weight with
its scales. The [FP8 processor](https://github.com/modal-projects/miles/blob/9d047db23a5760414de12185b3a133363ca2fde8/miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py)
already emits E4M3 weights and block scales, with architecture-specific selection
and multiple kernel/scale paths. Reuse these operations before writing a new
packer. Explicitly select and validate scale format: current Miles decisions can
also depend on environment variables and serving backend choices.

## Recipe interface

Add one flat field, `quantization`, with initial values `None` and `"fp8"`.
It selects the **serving base-weight representation**; it does not change training
precision. `None` preserves existing behavior, including checkpoint-defined formats.
Existing model-specific quantized recipes must not be relabeled as BF16 by default.

```python
from spindle.configs.qwen35_9b_lora_16k import Config as QwenConfig


class Config(QwenConfig):
    name = "qwen35-9b-lora-16k-fp8"
    quantization = "fp8"  # Proposed: validated blockwise serving profile.

    # Separate, optional training-compute experiment. These backend options
    # require validation with Miles' multi-LoRA layers before enabling a recipe.
    overrides = {
        "miles_cfg.cli_options.fp8": "e4m3",
        "miles_cfg.cli_options.fp8_recipe": "blockwise",
    }
```

Without the overrides, training keeps its current precision. Keep detailed
training options in `miles_cfg.cli_options` or the Megatron configuration, and
serving kernel choices in `sglang_cfg`/`inference_env`. Do not introduce a parallel
hierarchy of public training, transport, and inference configuration objects.
Backend-specific option spellings remain owned by those backends.

Resolve the shared serving format once during deployment. A small internal record
contains format/schema version, block shape, scale convention/dtype, activation
scheme, and the resolved quantized/retained module sets. Generate the export
configuration and SGLang checkpoint metadata from this record. Include its digest
in the base artifact identity and publication metadata. Overrides that disagree
with it must fail before GPU allocation; kernel choices that preserve it remain
allowed. `sglang_cfg.quantization` alone must not bypass the export contract.

The digest also includes the source checkpoint revision and conversion revision.
A replica must match the requested base artifact identity before loading an adapter.
Keep `model`/tokenizer identity separate from the generated inference asset path:
training still loads its intended checkpoint, not an accidentally substituted FP8
serving artifact. Resolve this inside asset preparation rather than changing the
meaning of the existing `model_weights` field for both consumers.

## Export and publication

### LoRA: prepare the base once

Prepare an immutable FP8 base artifact from the selected checkpoint with the same
conversion profile used by the deployment. Reuse a prequantized checkpoint only
when its metadata and source identity satisfy that profile. Publish the artifact
only after all tensor files and metadata are durable. Autoscaled replicas load
this artifact; they do not independently requantize a BF16 base.

Keep A/B adapter tensors, PEFT metadata, and the existing adapter publication path
unchanged initially. Each adapter version identifies its required serving base.
Do not merge adapters into the shared base: doing so changes the policy for other
clients and defeats multi-LoRA serving. A future merged-adapter optimization would
need a separate version and cache lifecycle.

BF16 training against an FP8 inference base creates a policy mismatch. Measure it;
do not describe this mode as quantization-aware training. If it harms learning,
use a separately validated trainer representation matching the frozen quantized
base, or add explicit fake-quantization/QAT support. Enabling TE FP8 matmuls alone
does not guarantee the same effective model as the serving quantizer.

### Full weights: convert before persistence, after HF mapping

Use this sequence for each immutable captured version:

```text
training weights at optimizer step N
  -> backend gather and HF name/layout mapping
  -> eligible weights quantized, matching scales emitted
  -> detached canonical snapshot
  -> persistence and bulletin publication
  -> replica staging and rank-local loader processing
  -> activation of the complete version
```

Quantize after gather/mapping so a block has the same boundaries regardless of
trainer TP/EP layout. A shard-local optimization is valid only when it reproduces
those canonical blocks and scales. All ranks still participate in the required
collectives; materialize/pack only on designated writers where possible. Avoid
quantizing redundant DP copies or gathering an entire model onto one GPU.

For Miles, reuse `HfWeightIteratorBase` and its quantized units. Spindle's full-weight
backend currently uses a different Bridge path: add a narrow iterator adapter at
that export boundary, sharing the conversion primitive where dependency-compatible.
Do not import the Miles GPU runtime into a Megatron-only process merely to reach a
quantizer. Factor the required primitive into a small dependency or implement a
compatible adapter there. This proposal does not require adding Miles full-training
support to Spindle.

A canonical snapshot contains HF names, logical shapes, real tensor dtypes, FP8
payloads, FP32 scales, and retained BF16 tensors. Transport bytes without numeric
casting; preserve each tensor's interpretation in the manifest. Backend-specific
FP8 padding, transposes, and packed kernel layouts belong to SGLang's loader, not
the control plane or the portable snapshot.

Keep weight/scale conversion groups together within bounded export buckets.
A changed scale makes its associated weight group dirty even if the FP8 bytes
happen to be unchanged. Inference must never observe new scales with old weights.

### Bootstrap and deltas

First establish full FP8 snapshot publication and reload. Version zero must be the
exact canonical quantized checkpoint that serving loads, including scale tensors
and exclusions. Do not delta FP8 bytes against the original BF16 checkpoint.

Then extend the current byte-delta path to the fixed quantized schema. Compare
canonical quantized bytes, not floating-point differences. Preserve logical dtype
metadata even if the compression container uses uint8. Reuse checksums, immutable
version manifests, and the publication lifecycle. A format, tensor-set, or shape
change starts a new full base; never reuse an incompatible parent snapshot.

Benchmark compression: moving block scales can change many weight bytes and make
FP8 deltas less compressible than BF16 deltas. Fall back to complete groups or a
full snapshot when delta size/work exceeds the full representation. Fewer bytes
per element alone does not establish lower publication latency.

### Async ownership and failure behavior

Complete all reads of live trainer tensors before releasing the captured version.
Quantization may run during capture, or afterward against detached storage; it
must not race the next optimizer step. Use bounded buffers/events, not unbounded
queued GPU copies. Persistence can overlap subsequent training once it owns an
immutable snapshot. Other LoRA clients continue independently.

The bulletin advertises a version only after payloads and metadata are durable.
On export failure, retain the last published version and preserve collective
participation/error propagation so peers do not deadlock. Never publish a subset
of weight/scale groups or reuse an incomplete artifact.

## Serving and training integration

Construct SGLang with the resolved FP8 configuration and matching base artifact.
Validate every supported model's quantized and retained module sets, including
fused QKV/gate-up mappings, MoE experts, embeddings/head, and hybrid-attention
projections. Reject unsupported combinations rather than silently converting
additional modules. FP8 base plus LoRA is a capability to test per model, GPU,
and linear/MoE backend, not a universal consequence of `enable_lora=True`.

For full updates, extend/test the existing Stitch stage/activate lifecycle. Native
SGLang loaders must consume both weights and scales, perform required local
postprocessing once per staged version, and preserve live parameter addresses or
recapture CUDA graphs when required. Activate only when every rank is ready.
Readers retain the old version on a preparation failure; a partial activation
failure fences the replica until a consistent version is restored. Invalidate
prefix caches on a base change and keep adapter-version cache isolation. No extra
controller or conversion work belongs on the token-generation request path.

FP8 training compute is a second validation track. Use Miles/Megatron/Transformer
Engine's supported recipe and verify LoRA forward/backward, expert layers, packed
sequence alignment, recomputation, and TP/CP/EP behavior. Keep trainable adapter
parameters and optimizer masters at their intended higher precision. Account for
recipe state in checkpoint/resume. Persistent FP8 parameters and FP8 parameter
gather are deferred: they require a separate master-weight and optimizer-init
review and cannot be enabled by the serving `quantization` field.

## Implementation sequence and acceptance

| Change | Acceptance before expanding scope |
| --- | --- |
| Resolve configuration and generate base artifact | Existing recipes unchanged; conflicting overrides rejected; startup metadata and export profile agree |
| FP8 base with BF16 LoRA updates | Cold/warm load, multiple adapters, unload/reload, independent publications, and restart all work on one dense model |
| Full FP8 snapshot reload | Weight/scale pairs reach every rank; live reload agrees with a fresh process loading the same artifact |
| Quantized deltas | Reconstructed bytes match full snapshots across versions; missing/corrupt groups fail without activating mixed state |
| FP8 training compute | Finite gradients, adapter updates, packed/recomputed execution, and checkpoint/resume validated before long RL |
| MoE and more hardware | Repeat loader, training, replay, and parallelism checks; approve explicit capability combinations |

Compare BF16 training/BF16 serving, BF16 training/FP8 serving, and FP8 training/FP8
serving with identical task streams, sample budgets, LoRA targets, and allocation.
Begin with a short dense-model check, then a MoE model to exercise expert routing.
For an async multi-client RL run, preserve staggered independent client loops and
record publication version on every rollout turn. Test sampling-support and router
replay explicitly; replay does not remove quantization-induced logit differences.

Report export/gather/quantization time, capture stall, peak HBM, transferred and
persisted bytes, stage/activation latency, time until a sampler is usable, output
TPS, and end-to-end client update time. Separately report sampler/trainer logprob
differences, PPO ratio/clip statistics, finite-gradient checks, and held-out reward
across repeated seeds. Compare scoring at identical weights before attributing
mismatch to async policy lag. Set numerical tolerances from the BF16 baseline and
quantization error measurements, not an assumed bitwise equivalence.

The first implementation should use Miles' existing conversion kernels. Only add
fused packing or overlap after profiling identifies quantization or copies as the
bottleneck. The design is modular at the existing backend-export, publication, and
serving-loader boundaries; it does not need a new public plugin framework.
