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

The 36-request forwarding budget is not a guarantee of 36 executing sequences:
adapter-slot, KV-memory, and prefill constraints can leave forwarded work waiting
inside SGLang. Evaluation therefore records native running and waiting counts
separately from sidecar admission. Autoscaling is not established by these
fixed-size-pool comparisons.

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
permanent adapter home or fixed replication factor. A replica merged with another
adapter remains eligible. Its backlog estimate is lower-bounded by the longest
observed remaining decode in that merged cohort, multiplied by the measured
merged token time, and it pays restoration time. The selected owner drains and
restores before serving the new adapter. This lets mixed demand reclaim a busy
specialized engine when that beats waiting behind an overloaded mixed engine.
The initial implementation excluded such engines while any mixed destination
existed. An early live RL trial exposed stranded demand, so that rule was removed
before restarting the measured comparison from the same initial checkpoints.

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

This adapts [dLoRA's](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
merge/batching and demand-driven replication ideas to local
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

Pool and live-RL comparisons are pending. The inference/deployment regression
suite passed 303 tests with one skipped before the reclaim correction. Real-engine checks on both one H200
(TP1) and two H100s (TP2) passed exact greedy-output and input-logprob restoration,
automatic specialization for 36 requests, and restoration before another adapter.

For 36 requests, 225 input tokens and 256 forced output tokens, warm H200 trials
took 6.97–7.14 seconds unmerged and 5.56–5.57 seconds merged. TP2 H100 trials took
5.21–5.32 seconds unmerged and 4.39–4.49 seconds merged. The initial unmerged trial
was slower on both configurations and is retained in the calibration artifacts.
First-use merge costs were 26.7 seconds on the H200 and 6.22 seconds on TP2 H100;
warm merge/restore costs were approximately 0.93/0.69 and 0.55/0.43 seconds,
respectively. These are microbenchmarks, not pool-level or RL improvements.

Every replay arm receives common full-batch unmerged/merged warmup and cache reset
before measurement. The service profiles describe that warmed regime. A cold
production replica cannot assume the warm transition cost on its first merge;
the profile's `initial_merge_seconds` charges the measured first-use cost until
one merge succeeds. Calibration emits this separate value. Evaluation's common
preparation pays that cost before measurement, identically in every arm.
The measured pool images precede the optional first-use guard; their source
hashes and engine commit are recorded in `deployed-source-hashes.json`. The guard
has a separate regression test. Warm comparisons do not validate its cold-start
behavior on a production pool.

The live comparison runs two optimizer updates per client with one prefetched
rollout. Both measured rollouts may therefore use the initial policy. It tests
real sampling, filtering, gradients, optimizer steps, and publication, but does
not establish updated-policy sampling. The replay experiments separately test
new exact versions. These are distinct claims.

Artifacts are recorded under `analysis/dlora-20261015`. The earlier eight-H200
attempt with retained pre-update replicas is excluded. Later H200 pool attempts
could not obtain the complete allocation. The replay study uses four TP2 H100
replicas (eight GPUs); live RL uses four TP1 H200 replicas and a four-H100 TP4
trainer. Initial live setup failures from environment JSON escaping and selection
of an incompatible initial checkpoint set are preserved, followed by a corrected
run using the four-GPU checkpoint set. Final tables must identify complete,
failed, and interrupted runs explicitly.
