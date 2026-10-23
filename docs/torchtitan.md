# Experimental TorchTitan multi-LoRA backend

This backend runs a native TorchTitan Qwen decoder with a shared frozen BF16
base and independently optimized LoRA adapters. It is an experimental integration:
GPU execution, 27B training throughput, and SGLang sampler parity have not yet been
validated. Do not interpret CPU correctness tests or model construction on the
meta device as successful 27B training.

The runtime pins TorchTitan to
[`58350410517dd62a15026e6816be874d0d5cb548`](https://github.com/pytorch/torchtitan/tree/58350410517dd62a15026e6816be874d0d5cb548)
and PyTorch to `2.16.0.dev20261023` (CUDA 13.0). Stable PyTorch 2.14.1 cannot
import this TorchTitan revision because it lacks a pipeline-liveness API.

## Recipe

Use `src/spindle/configs/qwen38_27b_torchtitan_lora_16k.py` as the initial recipe.
Ordinary `BaseConfig` fields select `backend = "torchtitan"`; backend options
live in `torchtitan_cfg` and support the existing dotted recipe overrides.

The initial implementation supports **one GPU per trainer instance**. Multiple
instances can be deployed using existing Spindle placement. Tensor, context,
pipeline, and fully sharded training within a trainer are not implemented.
The 27B recipe requests one H200 and admits up to eight clients; this allocation
is a candidate to validate, not a measured capacity guarantee.

Supported architectures are dense Qwen3.5 (0.8B, 2B, 4B, 9B, 27B) and Qwen3.8-27B,
using text inputs and unquantized HF safetensors. The loader checks architecture
settings, streams weights, excludes the vision tower, and checks that every
native parameter was loaded. MTP weights are not used. The public 4B and 27B
model configs have been checked against the native architectures.

## Adapter execution and persistence

The existing Spindle engine continues to select compatible ready requests and
serialize conflicting operations. This backend introduces no cross-client step
barrier. Each accepted model has its own seed, FP32 LoRA parameters, gradients,
AdamW state, and update counter. Forward-only requests leave gradients unchanged;
training accumulates gradients until that client's optimizer step.

Each microbatch pads sequences to a common length and supplies document boundaries
to both full attention and Gated DeltaNet. The base projection runs once for the
combined batch, while batched matrix multiplications apply the selected adapters.
Adapter weights are gathered per sequence, avoiding a copy per token. The fused
MLP retains its base gate/up projection and has independent adapter matrices for
each branch. It exports those branches as ordinary PEFT `gate_proj` and `up_proj`
weights. Both targets must be selected together for the fused MLP.

Activation checkpointing is enabled by default. The frozen vocabulary projection
uses chunked, recomputed log probabilities to avoid retaining a full token-by-vocab
logit tensor. Layer compilation is optional and has not been benchmarked.

Supported losses currently follow the existing Miles input adapter: cross entropy,
importance sampling, PPO, CISPO, and DRO. Router replay, restricted sampling masks,
GDN projection adapters, unembedding adapters, MoE, and multimodal inputs are not
supported. Client training flags must match the deployment's target modules;
`train_unembed=False` is required.

Training checkpoints capture detached CPU copies. Checkpoints with optimizer state
also include accumulated gradients so a save between backward and optimizer step
can resume correctly. Sampler captures contain detached PEFT weights and publish
through the existing snapshot bulletin. Checkpoint formats are specific to this
backend and cannot be loaded as Miles optimizer checkpoints.

## Validation

Local validation on 2026-10-23: **153 tests passed**, covering the new backend,
deployment configuration and CLI, Modal deployment setup, and engine regression
tests. Ruff and `git diff --check` also passed.

Local tests cover batched/serial output and gradient parity, repeated adapter IDs,
optimizer isolation, accumulation, forward-only behavior, nonfinite gradients,
checkpoint resume, immutable captures, fused-MLP exports, native 27B construction,
and native HF weight conversion (including tied embeddings and split shards).
Model construction uses meta tensors and does not allocate 27B weights.

GPU validation is pending. The commands below target the existing Modal
`lilo-deploy` environment and have not completed a GPU run. Run them sequentially and
stop on a failure; neither launcher retries a failed or OOM run automatically.

From the worktree root, using an environment with the repository dependencies:

```bash
PYTHONPATH=src modal run --env lilo-deploy scripts/probe_torchtitan.py
PYTHONPATH=src modal run --env lilo-deploy scripts/modal_torchtitan_benchmark.py \
  --model Qwen/Qwen3.5-4B --flavor 4B --output torchtitan-4b.json
PYTHONPATH=src modal run --env lilo-deploy scripts/modal_torchtitan_benchmark.py \
  --model Qwen/Qwen3.8-27B --flavor 27B --output torchtitan-27b.json
```

The probe has a 20-minute GPU timeout; each model benchmark has a 50-minute GPU
timeout. These three calls therefore have a maximum combined allocation of two
H200 GPU-hours. Each call releases its GPU on completion. The benchmark downloads
public model weights into its ephemeral container; it does not modify an existing
model-assets volume or shared deployment.

Default benchmark settings: sequence length 1024; 1, 2, 4, and 8 adapters; rank 16;
two warmup steps and three measured steps per adapter count. The runner checks
packed-versus-serial log probabilities and gradients after training, detached
exports, successful updates, packed and serial training throughput, and peak CUDA
memory. It records the exact runtime and
configuration in JSON. A failed run produces no successful benchmark report.

This is a **synchronized packed-backend benchmark**. It does not establish
asynchronous client throughput, rollout/publication latency, inference parity, or
an advantage over Miles. A follow-up end-to-end test must use the ordinary SDK
path with independent staggered client loops and record each client's rollout,
training, and publication timestamps separately.
