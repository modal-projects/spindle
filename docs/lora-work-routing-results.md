# Measured-work routing experiment

This experiment tests a simpler alternative to an additive queue/prefill/decode/
adapter/eviction score. Results are pending real-GPU validation. It is implemented
as `inference_lora_routing = "work"`; the existing default is unchanged while it
is evaluated. The comparison baseline is native Flash KV affinity.

## Policy

Each sampling process owns one router for the whole inference pool. Every
destination uses the same measured service profile, in seconds, for the deployed
model, adapter rank, hardware, and tested context/concurrency regime:

```
GPU backlog = outstanding uncached prefill / prefill rate
            + outstanding remaining decode / aggregate decode rate
            + one GPU transfer per pending nonresident adapter
CPU readiness = one registration per distinct queued missing CPU adapter

score = max(GPU backlog, CPU readiness)
      + new GPU transfer, if not already resident or pending
      + this request's uncached prefill / prefill rate
      + this request's expected decode / aggregate decode rate
```

This estimates projected drain work, not exact request completion time. CPU
registration can overlap existing GPU work, so those clocks combine with `max`.
There are no dimensionless tuning weights, speculative future eviction costs,
fixed affinity bonuses, or autoscaling-target multipliers. For the fixed-length
workload, output length is known. Production currently uses the requested output
limit, so early-EOS distributions need additional validation.

The lowest score wins. Numerical ties retain the previous destination, otherwise
version/prefix rendezvous hashing resolves them deterministically. Selection and
reservation are synchronous on the frontend event loop. A group hashes its
prompt once. Every physical generation reserves its own expected work before
network I/O; release occurs on success, error, or cancellation. Observed request
IDs prevent counting a local reservation twice after the replica reports it.
Sibling requests share the adapter-setup estimate, but receive no speculative
credit for another request's unfinished prefill.

The current model counts one transfer per distinct queued nonresident adapter.
It does not predict repeated reloads if more adapters interleave than fit in GPU
slots. This is a testable approximation, not an eviction-cost model. Registration
uses a conservative full-service estimate, including earlier distinct queued
registrations, until the next observation; it does not know byte-level transfer
progress or whether the Modal Volume filesystem already cached the file.

## Observations and request path

The pinned SGLang scheduler exports a snapshot once per second. It inspects CPU
metadata only: no tensor contents, CUDA synchronization, or synthetic cache
lookups. A bounded background writer publishes snapshots through a local atomic
file. The sidecar exposes them through authenticated Flash `/routing-state`, and
each frontend polls off the selection path. Cold-start refresh remains coalesced
and bounded; the work policy does not make native-affinity lookup RPCs.

Snapshots include actual CPU/GPU adapter inventories, remaining request work,
and hashed reusable prefix boundaries scoped to the exact registered adapter.
Hybrid-model cache credit requires both the full KV path and a resident Mamba
state at the matched boundary. Host-only state, unknown cache implementations,
salted namespaces, and bigram keys receive no credit. Snapshots replace previous
sets, removing evictions and avoiding event-loss recovery. Transport is bounded
to the 8,192 most recently accessed supported boundaries per replica. Hashes use
64-token boundaries and exclude the last prompt token, whose logits must be
computed. This conservatively misses some reusable nonaligned prefixes.

The initial implementation supports one data-parallel replica and one pipeline
stage per container. Missing or stale engine snapshots fall back to native
Flash. Backend cache and exact-weight-version enforcement remain authoritative.
Snapshot freshness and reservations are approximate distributed observations;
they do not impose a global queue limit.

## Planned validation and measurements

- Eight H200 inference replicas, Qwen/Qwen3.8-27B, TP=1, rank-32 adapters,
  eight GPU adapter slots and 256 CPU registrations; at most eight experiment
  GPUs active. Requested running capacity 128 is checked against actual engine
  capacity. The fixed pool isolates routing from autoscaling.
- A serialized 8K repeated-prefix control checks native reuse and greedy output
  hashes before load testing. Short per-replica calibration measures uncached
  prefill, 36-request decode throughput, registration exposure, and paired
  GPU-cold/GPU-hot adapter latency with equal KV hits.
- Sixteen independent replay clients on eight actual CPU frontend containers,
  staggered starts, no inter-round barriers. Each client's next rollout overlaps
  its recorded update/publication delay. Five-minute arrival windows stop new
  rollouts at the deadline and measure the final drain separately.
- Warmed long-prefix reuse and new-version arrival cases use 8,192 input tokens
  (7,168 shared + 1,024 varying suffix) and exactly 512 output tokens with
  `ignore_eos=True`. Native-affinity serialized warmup precedes measurement.
- Version turnover uses saved immutable trained checkpoints that are initially
  unregistered on inference replicas. It exercises real adapter registration,
  but does not claim guaranteed network-cold Modal Volume reads or new optimizer
  work. Independent optimizer/publication timing is replayed, not executed.
- Report request/rollout latency, completion throughput, token-weighted KV hits,
  first-prefill hot-LoRA hits, GPU loads/evictions, CPU registrations, retries,
  per-replica load, routing CPU time, snapshot overhead, and version correctness.
  Compare measured-work routing, native KV affinity, and an adapter-cost ablation.

The earlier native-home results remain in
[the preceding report](lora-kv-training-replay-results.md). They are not results
for this policy.
