# Decentralized merge-aware LoRA serving

Status: implementation and validation in progress. No throughput improvement is
claimed until the paired inference and live RL comparisons finish.

## Scope and dependencies

The implementation uses SGLang PR #44354 at
`708d38f143697e25c3e687c20fc0b043930c8416`, with `lora_unmerge_mode=restore`.
Original affected weights are retained on CPU. Subtraction is not used because
repeated subtraction can accumulate rounding error. Both transitions drain
requests and flush prefix/recurrent state. Merged and unmerged arithmetic may
produce different samples; exact restoration is tested separately.

The existing separation between training and inference is preserved. Stitch
version identities and bulletin publication still select immutable exact adapter
versions. This is not prefill/decode disaggregation (the PR currently rejects
that execution mode). Active generations are not migrated between engines.

## Ownership and request path

Each sampling container has one router per inference pool, with asynchronous
replica observation and local reservations. Selection is synchronous over the
snapshot. Neither request selection nor replica mode selection requires a global
controller, distributed lock, or per-request control-plane write.

Each replica sidecar has one local mode controller. The event loop owns its
bounded admission count and pending queue. At most `capacity + queue_buffer`
requests are admitted, with at most `capacity` forwarded concurrently. The
sidecar overflow buffer is four requests. SGLang allows a 40-request handoff
because it counts arrivals before scheduling, including free running slots.
The sidecar forwards at most 36 requests, so this does not add a second
40-request backlog. A native limit of four caused spurious rejection of work
that fit available execution slots in the admission diagnostic.
Capacity rejection is distinguished from an engine failure and retried with
short randomized delay. Exact adapter constraints survive every retry.

The sidecar serializes only transitions on its own engine. It stops incompatible
refill, drains active work, restores the original weights, and then serves other
versions/adapters. A failed or ambiguous transition fails closed and makes the
sidecar health check fail. Client cancellation releases local admission; active
generation cancellation uses the existing engine abort path.

## Routing

The first implementation retains the measured-work router's version-scoped
token-prefix observations and local reservation accounting. It compares queued
service plus remaining prefill/decode work in seconds. A measured merged decode
rate replaces the unmerged rate on a matching specialized replica. This is an
approximate service model, not an optimal completion-time estimator.

A bounded group route expires when its observed plus locally reserved demand
reaches engine capacity. Subsequent groups can use any replica; there is no
permanent adapter home or fixed replication factor. A busy replica merged with
another adapter is excluded while compatible destinations exist. If every
replica is specialized, new versions remain eligible: the selected owner drains
and restores before serving them. Idle specialization is reclaimable.

Waiting for a warm prefix remains a valid choice. There is no rule that an idle
cold replica always wins. Prefix observations are discarded after transitions;
snapshots predating the transition cannot advertise those invalidated prefixes.

## Local merge plan

At an idle decision boundary, compare two plans for admitted work: a mixed
unmerged batch versus a homogeneous merged batch followed by the minority work.
The objective is estimated sum of completion times for that local batch.
The candidate adapter is the largest pending cohort. The specialized plan pays
merge time, restoration time, and full candidate-prompt prefill after cache
invalidation. It receives no credit for future requests that have not arrived.
Partial batches receive proportionally less decode-saving credit than the
calibrated full batch. The oldest minority's waiting time acts as a fairness
credit: a new specialization cannot bypass it after the configured wait bound.
Once incompatible demand arrives on a merged engine, refill stops and it drains.

This adapts dLoRA's merge/batching and demand-driven replication ideas to local
ownership. It does not reproduce dLoRA's active request/KV co-migration or claim
equivalent optimality. Important limitations are the approximate batch service
model, conservative prefix-rebuild accounting, and transitions only after drain.

## Evaluation protocol

All compared policies must use the same pinned engine, GPU layout, admission
bounds, model, rank, initial adapters, and workload seed. Include native KV
affinity, the previous simple overflow control, this router with merging disabled,
and this router with merging enabled. Count retries and transition time in the
measured interval. Do not compare this engine directly against old-engine runs
as if only routing changed.

Retain the existing 64-version long-prefix serving trace, independently staggered
RL-shaped bursts with version turnover, and multi-turn continuation workloads.
Live RL uses independent client rollout/train/publication loops and common
initial checkpoints. Report actual generated and trained tokens, filtering,
per-client update latency, reward/output-length changes, and publication count;
equal seeds do not ensure equal generated trajectories across execution modes.

Collect request latency and KV hits, GPU adapter loads/evictions, actual executed
decode-batch size and adapter composition, merged decode-step fraction, mode
transition durations, per-replica running/waiting demand, and rejection counts.
Resource monitors stop only task-owned apps and preserve allocation evidence.

## Results

Pending. Local regression tests and real-engine validation artifacts are recorded
under `analysis/dlora-20261015`; final tables must identify complete versus failed
or interrupted runs explicitly.
