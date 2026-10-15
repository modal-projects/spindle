# Decentralized merge-aware LoRA serving

Status: experimental, opt-in implementation. The initial pool, revised admission,
and live RL comparisons are complete; additional burst workloads are under test.
Trained-adapter BF16 agreement passes the merge PR's tolerance on twelve generated
trajectories but fails on the original prompt tokens. These are different tests;
neither establishes numerical agreement for arbitrary requests or batches.

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

The sampling API and payload remain unchanged. Activation is explicit:
`inference_lora_routing="dlora"`, measured `inference_routing_work_rates`, the
replica profile in `inference_env["SGLANG_SPINDLE_MERGE_PROFILE"]`, and
`sglang_cfg.enable_lora_merge=True` with `lora_unmerge_mode="restore"`.
The deployment selects the pinned merge-capable image only when that engine
option is enabled. Existing default deployments retain their current image and
policy. The profile must be calibrated for the model, rank, batch size, and
hardware; benchmark setup endpoints are confined to the instrumented test image.

## Ownership and request path

Each sampling container has one router per inference pool, with asynchronous
replica observation and local reservations. Selection is synchronous over the
snapshot. Neither request selection nor replica mode selection requires a global
controller, distributed lock, or per-request control-plane write.

During active demand each frontend polls replica observations approximately once
per second, concurrently across hosts; Flash membership is refreshed every five
seconds. One in-flight refresh is shared by concurrent callers. Fresh selections
do not await it. The initial/stale burst may await that one refresh, bounded at
three seconds; unavailable observations fall back to Flash. Work snapshots older
than three seconds are excluded. This decentralizes decisions, but observation
traffic still scales with frontends times replicas.

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
The final sidecar also counts latest-version resolution against this admission
bound before awaiting checkpoint I/O. A regression test blocks resolution,
checks rejection before registration, cancels a waiter, and verifies that its
capacity is released. This guard was added after freezing the measured GPU
images; the replay requests already specify exact versions and do not enter
that pre-admission resolution path.

The 36-request forwarding budget is not a guarantee of 36 executing sequences:
adapter-slot, KV-memory, and prefill constraints can leave forwarded work waiting
inside SGLang. Evaluation therefore records native running and waiting counts
separately from sidecar admission. Autoscaling is not established by these
fixed-size-pool comparisons.

The revised admission policy also respects `max_loras_per_batch`. Running
adapters reserve slots; oldest pending adapters reserve the remaining slots.
Only four incompatible requests may wait, preserving admission capacity for
compatible siblings. After five seconds of waiting, an incompatible adapter
stops compatible refill until a slot drains. This is a bound on bypass, not a
five-second completion guarantee: running generations are not preempted.
The router observes these reservations and estimates the time until one cohort
releases a slot. Admission remains authoritative on each replica; observations
and frontend reservations are approximate and can still cause rejected retries.

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

For a group chunk of `q <= capacity` samples, the base score is
`backlog + registration + GPU_load + q * (uncached_prefill / prefill_rate +
requested_decode / aggregate_decode_rate)`. Backlog includes observed outstanding
tokens plus this frontend's requests absent from the observation. Prefix hits
must match the exact adapter version and a token-prefix boundary advertised by
the engine. Slot-release and incompatible-merge drain times lower-bound backlog;
restoration time is added when needed. All terms have units of seconds and use
measured rates. The minimum score chooses the destination. A planned sibling
chunk stays together only while its destination has admission capacity.

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

The KV control uses Flash's session-affinity header and the existing sample
retry path. A capacity rejection changes the physical retry affinity while
retaining the logical conversation identity. It is therefore an affinity policy
with overflow, not a baseline that waits forever on one replica. The custom
router uses exact-upstream headers after stripping conflicting Flash affinity
headers; native request payloads still retain their logical cache identity.

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

The revised inference/provider/deployment regression suite passed 408 tests with
one skipped. Full CI passed 916 tests with one skip on Tinker 0.24.1 and 911
with six skips on 0.25.0. Real-engine checks on both one H200
(TP1) and two H100s (TP2) passed exact greedy-output and input-logprob restoration,
automatic specialization for 36 requests, and restoration before another adapter.

For 36 requests, 225 input tokens and 256 forced output tokens, warm H200 trials
took 6.97–7.14 seconds unmerged and 5.56–5.57 seconds merged. TP2 H100 trials took
5.21–5.32 seconds unmerged and 4.39–4.49 seconds merged. The initial unmerged trial
was slower on both configurations and is retained in the calibration artifacts.
First-use merge costs were 26.7 seconds on the H200 and 6.22 seconds on TP2 H100;
warm merge/restore costs were approximately 0.93/0.69 and 0.55/0.43 seconds,
respectively. These are microbenchmarks, not pool-level or RL improvements.

### Initial long-prefix serving comparison

Four replicas, each two H100s; Qwen3.8-27B, BF16, rank 32 on all seven configured
linear targets, 16,384 context limit, 36 running requests, eight GPU adapter slots
and 256 CPU adapter registrations per engine. The driver has eight independent
router instances and 512 outstanding streams: 64 exact adapter versions, eight
streams per version, four requests per stream, 2,048 requests total. Each request
has 8,192 input tokens (7,168 shared) and 512 forced output tokens. All arms use
the same seed, adapters, HTTP/1.1 transport and common warmup/cache reset.

| Policy | Wall seconds | Output tokens/s | Mean / p95 request seconds | KV token hit | GPU LoRA hit | GPU loads | Mean decode batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| KV affinity | 644.48 | 1,627.01 | 122.36 / 310.23 | 14.72% | 34.77% | 1,336 | 11.32 |
| Work router, merge disabled | 638.04 | 1,643.42 | 119.96 / 307.03 | 20.55% | 33.89% | 1,345 | 11.10 |
| Work router, merge enabled | 624.45 | 1,679.19 | 119.77 / 291.29 | 18.82% | 36.33% | 1,296 | 11.60 |
| Model placement with overflow | 402.18 | 2,607.23 | 75.09 / 176.98 | 48.37% | 61.23% | 660 | 19.90 |

These are single trials, not confidence intervals. All arms executed zero merged
requests during measurement. Enabling merge therefore does not explain its
3.2% throughput difference from KV affinity. The simple overflow control is
60.2% faster than KV affinity and materially outperforms the first work router.
There were no transport errors. Capacity rejections were respectively 74,567,
66,770, 65,388 and 33,866, including retries in measured request latency.

Native batches contained about 7.6–7.9 distinct adapters against the limit of
eight, while executing only about eleven requests. Thus forwarding 36 requests
did not fill a 36-sequence decode batch. The revised admission policy targets
this observed batch constraint. GPU copy measurements alone do not explain the
result: the KV arm accumulated about 75 host-seconds of loading across four
replicas during 644 wall-seconds, and that counter is not critical-path time.

### Revised adapter-slot admission

A new four-replica pool repeats the same serving configuration with the
adapter-slot gate and slot-release estimate enabled only for the new policy.
All three arms disable merging during measurement. Per-replica source hashes and
the pinned engine revision were verified. The controls retain the same overall
36+4 admission bound; only the new policy reserves native adapter slots.

| Policy | Wall seconds | Output tokens/s | Mean / p95 request seconds | KV token hit | GPU LoRA hit | Loads / evictions | Mean decode batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| KV affinity | 643.56 | 1,629.34 | 120.17 / 310.02 | 14.97% | 36.52% | 1,295 / 1,291 | 11.45 |
| Model placement with overflow | 391.16 | 2,680.67 | 73.37 / 175.23 | 49.46% | 60.79% | 657 / 653 | 20.41 |
| Revised adapter-slot policy | 418.55 | 2,505.26 | 72.09 / 266.77 | 49.49% | 77.29% | 424 / 420 | 18.43 |

The revised policy improves throughput by 53.76% over KV, reduces mean latency
by 40.01%, and reduces adapter loads by 67.26%. However, it remains 6.54% slower
than model placement with overflow, with 52.24% higher p95 latency. Its 35.46%
fewer loads than overflow do not compensate for smaller native batches and the
long tail. Cache/load counters are explanatory measurements, not optimization
objectives on their own. Rejections are 74,190 / 34,831 / 30,554 respectively;
all 2,048 requests execute exactly once in each arm, with no transport errors.

The measured new-policy routing selection p50/p95 is 157.6/220.2 microseconds,
including the larger active reservation sets in this trace. Selection itself
does not make RPCs. A separate in-process snapshot benchmark measured approximately
49 microseconds median at four replicas, 80 at eight and 1,000 at 128; those CPU
measurements exclude discovery and do not establish serving scalability.
The first revised trial runs KV, model-placement overflow, then the revised
policy. The planned repeat swaps the two controls and still runs the revised
policy last; it is not a fully counterbalanced experiment. Common cache and
mode preparation is repeated before every arm.

KV hits count cached input tokens divided by input tokens at first native
admission. LoRA hits count requests whose exact adapter is already in a GPU
buffer before fetching it. Loads/evictions are logical per-replica adapter-buffer
events, observed on TP rank zero, not separate counts for every GPU shard.
Loading seconds are aggregate host elapsed time across replicas, not isolated
CUDA transfer time or exposed request delay.

### Additional fixed-work workloads

All three replays use four TP2 H100 replicas, the same 27B model and rank-32
adapters, eight separate CPU frontends, HTTP/1.1, and seed 15101. Clients start
0.5 seconds apart and advance independently. Each policy receives identical
initial CPU registrations and prefix warmup. Generation uses `ignore_eos=True`;
output lengths and request counts are checked, as are exact adapter versions
and finite logprobs. Native admission counts detect extra generation caused by
transport retries. Recorded update
delays are replayed alongside the next prefetched rollout; they are not new
optimizer work and do not impose a cross-client barrier.
The long-prefix and multi-turn GPUs are in `us-east`; the short-decode-study
GPUs are in `us-west`, while its frontends remain in `us-east`. Every policy
within that comparison uses the same cross-region path. Absolute latency across
these differently placed pools is not a controlled comparison.

| Trace | Clients / active versions | Logical concurrent requests, maximum | Requests / output tokens | Prompt / decode lengths | Arrival structure |
|---|---:|---:|---:|---|---|
| Long-prefix bursts | 16 / 16, with 32 versions over the run | 1,088 | 3,264 / 3,342,336 | 8,192 / 1,024 | Three rounds; hot client 128 samples, seven clients one 64-sample group, eight clients eight groups of eight |
| Multi-turn continuation | 16 / 16, with 32 versions over the run | 256 | 2,304 / 2,359,296 | 8,192 then 9,472 then 10,752 / 1,024 each turn | Three rounds, two groups of eight per client, three turns per trajectory |
| Short-prompt, long-decode | 12 / 12 | 480 | 960 / 3,932,160 | Actual short RL math prompts / 4,096 | Two rounds; hot client one 128-sample group, eleven clients four groups of eight |

Long-prefix and multi-turn rounds zero and one reuse version 4; round two uses
version 7, initially absent from the CPU adapter registry. The two-round short
trace uses version 4 throughout. Multi-turn prompts append each trajectory's
actual previous output and a deterministic 256-token synthetic tool reply.
Consequently their lengths and dependency graph match across policies, while
generated token contents can differ. This is a tool-shaped replay, not a claim
to execute a real external tool or measure task reward.

Generation makespan ends at the last response and includes routing/retry delay.
End-to-end harness wall time also includes returning detailed traces from CPU
frontends. Both are retained: the large long-prefix KV trace has a 13.48-second
collection tail. This instrumentation cost must not be mistaken for inference
service time. Per-request latency ends at its response and includes time spent
waiting outside the engine.

The completed long-prefix burst comparison is:

| Policy | Generation seconds | Output tokens/s | Mean / p95 request seconds | KV token hit | GPU LoRA hit | Loads / evictions | Mean decode batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| KV affinity | 824.52 | 4,053.68 | 133.76 / 256.67 | 62.00% | 76.90% | 570 / 566 | 26.32 |
| Model placement with overflow | 629.75 | 5,307.41 | 97.97 / 195.63 | 78.65% | 96.88% | 94 / 91 | 34.78 |
| Revised adapter-slot policy | 744.64 | 4,488.52 | 123.70 / 231.24 | 62.56% | 81.40% | 444 / 440 | 31.02 |

Relative to KV, the revised policy gains 10.73% throughput and reduces mean/p95
latency by 7.52%/9.91%. It still loses to the simple overflow policy. The latter
uses 2.75 replicas per exact adapter and 2.01 per prefix group, versus 3.97 and
3.65 for the revised policy. Its native batches contain 4.47 distinct adapters
on average, versus 7.42. The gate prevents exceeding eight adapter slots, but
does not capture the benefit of less heterogeneous batches and less replicated
prefixes within that bound. No busy observation has an idle replica in either
KV or the revised arm; the failure is not wholesale idle capacity.

Capacity rejections are 587,996 / 416,961 / 546,406 in table order. All arms have
exactly 3,264 engine admissions; KV has one transport error without a duplicate
admission, and the other arms have none. Mean time outside the engine is 99.25
seconds for KV and 94.65 for the revised router, compared with 34.51 and 29.05
seconds inside it. GPU load host-seconds are 30.89 / 5.55 / 24.94. Merging is
disabled for these three arms. These measurements identify batching, prefix
replication, and excess-demand retries as material limitations; fewer adapter
copies alone are an inadequate objective.

For the selected destination, the revised router's cached-token estimate exactly
matches native admission in 88.33% of the 64-version serving requests and 72.89%
of the larger burst requests. It underestimates in 9.33% / 26.07% and
overestimates in 2.34% / 1.04%, respectively. Mean absolute discrepancies are
579 / 1,492 tokens. These compare a selection-time estimate with a later engine
observation: intervening requests can populate or evict prefixes. They do not
measure counterfactual reuse on unselected replicas and do not establish that
the selected destination was optimal. The observed cache metadata is useful,
but accurate matches on many requests do not rescue the current placement rule.

The first revised-policy multi-turn trial has one such transport incident.
An accepted attempt appears in native observations from 466.03 to 496.27 seconds
of the trace, but its HTTP/1 response-header wait fails only after 360.06 seconds.
The retry succeeds on another replica, causing 2,305 admissions for 2,304 useful
responses. Useful generation makespan is 842.65 seconds (2,799.85 tokens/s);
mean/p95 request latency is 44.39/81.72 seconds. This is a retained operational
result, not a clean fixed-native-work comparison. The last other response arrives
at 569.42 seconds; removing the failed request is not used to manufacture a
replacement throughput result. A single repeat with a fresh driver and identical
settings is scheduled. The underlying transport failure is not attributed to a
specific proxy component, and production connection settings are unchanged.

### HTTP/1 connection admission

The initial short-prompt comparison exposed a separate client-side bottleneck.
HTTPX's default transport limit permits 100 concurrent connections per sampling
call. For the hot client's first 128-sample batch in the merge-enabled arm, the
first 100 attempts reach their first HTTP event within 0.173 seconds and all
complete in merged mode. The other 28 wait 63.81–64.03 seconds before their first
TCP-connect event. Their routing reservations were made before that wait, so
their placement observations were already obsolete when dispatched.

Those 100 merged requests complete in 67.40 seconds on average, with a maximum
of 68.83 seconds. The other 28 average 222.48 seconds and the final response takes
402.50 seconds. Mean latency for the 128-request batch improves from 126.19
seconds without merging to 101.33 with it, while its last response worsens from
326.78 to 402.50 seconds. The group-return API makes that tail visible to the RL
client. This is a demonstrated transport admission effect; it must not be
attributed entirely to the replica scheduling objective.

The final HTTP/1 sampling transport removes that hidden connection limit while
retaining TCP keepalive and the bounded replica scheduler. A real local HTTP
regression test holds every response until all 128 requests arrive: the old
100-connection limit times out, and the revised transport passes. A caller that
supplies its own client or transport still owns its connection settings. This
change does not remove HTTP/2's independently negotiated stream limit.

The four-policy short-prompt workload is being repeated with the fix applied
equally to every arm and the policy order reversed. Its model, adapters, forced
output lengths, client loops and GPU topology are unchanged. The original
results remain visible as measurements of the previous transport. The scheduled
multi-turn recovery also uses the corrected helper, but its sampling calls have
one or eight samples, below the old limit; that limit never binds there.

### Numerical validation

On two trained rank-32 version-4 adapters, fixed-prompt input logprobs differed
between merged and unmerged BF16 execution by mean absolute 0.1266 / 0.1290 and
maximum 1.5861 / 1.4355. Both exceed the pinned PR test's mean 0.03 / maximum 0.25
bounds. The first 32 greedy output tokens matched and restoration was exact;
neither establishes merged numerical equivalence. A zero-initialized adapter
passed with zero logprob difference. The tolerance originates in smaller-model
tests, but it has not been relaxed to make this experiment pass.

A second check uses actual RL math prompts (250 and 217 tokens), generates
2,048 tokens with each trained adapter, and teacher-forces those same tokens
through unmerged, merged, and restored execution. On the generated-token
positions, mean differences are 0.007731 / 0.007621 and maxima are
0.138362 / 0.120170: both pass the unchanged bounds. Restoration remains exact.
This check compares conditional logprobs on a fixed trajectory; it does not
establish that independently generated 2,048-token trajectories agree, nor does
it test numerical agreement in a 36-sequence batch. The original prompt-token
failure remains a separate result. The expanded check covers all twelve adapters
used by the short-prompt, long-decode replay: all pass on 2,048 generated tokens
each, with per-adapter mean differences 0.005944–0.011387 and maxima
0.109493–0.171487. Exact restoration passes for every adapter. This numerical
gate completed before starting that replay's four-policy comparison.

A read-only tensor diagnostic on layer 11's value projection found that BF16
merging leaves 99.27% of base-weight entries unchanged; the norm of the rounding
error is 99.29% of the LoRA update norm. This layer's largest update is only
2.50e-6. FP16 reduces the relative error to 95.05%, still large. These are
one-layer measurements, not an end-to-end attribution, but they demonstrate
that these small training updates are poorly represented in merged low-precision
weights. The fixed-prompt merged/unmerged discrepancy is also larger than the
unmerged adapter/base discrepancy (mean approximately 0.101).

A separate two-H100 FP16 diagnostic failed during engine startup, before serving:
the pinned hybrid-attention backend assigned Half activations to a BFloat16
convolution-state buffer. It was stopped and contributes no performance result.
Changing weight precision is therefore not a validated workaround in this setup.

### Live RL comparison

Eight clients staggered by five seconds completed two optimizer updates each on
four TP1 H200 inference replicas and one TP4 H100 trainer. Each update retains
four DAPO groups of eight samples, filtering all-correct/all-wrong groups, with
four-group refills and a 64-candidate-group budget. Generation permits 8,192
output tokens, thinking enabled, temperature/top-p 1.0, seed 4242. One rollout is
prefetched. Initial checkpoints are identical across the paired arms.
The optimizer uses learning rate 1e-5, Adam betas 0.9/0.95, epsilon 1e-12, gradient
clip norm 1.0, and asymmetric policy-ratio clipping at 0.8/1.28. Each inference
replica has eight CPUs and 192 GiB RAM; the trainer has 32 CPUs and 256 GiB RAM.

The completed KV baseline took 3,380.15 seconds for 12,295,732 generated and
4,057,999 trained output tokens: 3,637.63 generated tokens/s and 1,200.54 trained
tokens/s including training and publication. It sampled 228 groups, filtered
154, retained 64 and left ten unused mixed-reward groups. Mean output length was
6,741 tokens and mean reward 0.5691. Only 52 GPU adapter loads and 50 evictions
occurred, accumulating 3.62 host-seconds of loading. KV token hit ratio was
35.79%, GPU LoRA-buffer hit ratio 91.28%, and mean executed batch 24.77. This
workload does not exhibit substantial adapter-transfer overhead.

| Metric | KV affinity | Merge-aware router |
|---|---:|---:|
| Completed optimizer updates | 16 | 16 |
| Wall seconds | 3,380.15 | 3,520.49 |
| Generated output tokens | 12,295,732 | 12,789,845 |
| Generated tokens/s including training | 3,637.62 | 3,632.97 |
| Trained output tokens | 4,057,999 | 3,792,914 |
| Trained output tokens/s | 1,200.54 | 1,077.38 |
| Mean / p95 client update seconds | 1,347.63 / 2,419.37 | 1,460.18 / 2,525.25 |
| Sampled / filtered groups | 228 / 154 | 240 / 169 |
| Mean output length | 6,741.08 | 6,661.38 |
| Mean reward | 0.5691 | 0.5557 |
| KV token hit ratio | 35.79% | 33.88% |
| GPU LoRA-buffer hit ratio | 91.28% | 94.20% |
| GPU loads / evictions | 52 / 50 | 32 / 32 |
| GPU load host-seconds | 3.62 | 2.05 |
| Mean executed batch / distinct adapters | 24.77 / 3.91 | 23.91 / 4.67 |
| Merged decode-step fraction | 0% | 15.24% |
| Mode transitions / aggregate seconds | 0 / 0 | 14 / 16.65 |

There is no RL throughput improvement: generated tokens/s is 0.13% lower. Trained
tokens/s is 10.26% lower, combining 6.53% fewer retained output tokens with 4.15%
more wall time. Both arms retain 64 groups, but output lengths and DAPO filtering
vary with stochastic generation and batching. The new arm samples twelve more
groups and filters fifteen more. This is an observed end-to-end outcome, not a
controlled estimate of router-induced training slowdown. The fixed-work replay
is the corresponding scheduling control.

Fewer adapter loads save only 1.57 aggregate host-seconds; that cannot explain a
large throughput gain in a roughly one-hour run. The new arm has slightly smaller
and more heterogeneous native batches and lower KV reuse. Merged execution does
occur (120 requests in the engine observation window, including overlapping
initialization), but the benefit does not translate into faster overall sampling.
These live pools used the reclaim correction and the original admission
controller, before the optional adapter-slot gate. With eight active clients,
this workload does not exceed the eight-adapter native batch limit.

All initial policy adapters have zero-initialized LoRA B matrices. The numerical
zero-adapter check passes, but these two-update prefetched runs do not validate
merged inference on trained versions. The final client's DAPO filtering was
allowed to finish under a replacement bounded monitor; client processes,
workload and policy were unchanged. The completed run still finished within the
original one-hour budget. Observation coverage is 99.16% for KV and 96.00% for
the new arm; a 138-second observer-restart gap is excluded from time-weighted
statistics. Both live arms and their eight GPUs have been stopped.

Every replay arm receives common full-batch unmerged/merged warmup and cache reset
before measurement. The service profiles describe that warmed regime. A cold
production replica cannot assume the warm transition cost on its first merge;
the profile's `initial_merge_seconds` charges the measured first-use cost until
one merge succeeds. Calibration emits this separate value. Evaluation's common
preparation pays that cost before measurement, identically in every arm.
The initial pool and live images precede the optional first-use guard; the
revised pool includes it. Source hashes and engine commits are recorded in
`deployed-source-hashes.json`. A separate real-engine cold-start check merged all
36 requests with 4,096 forced output tokens, charging a 6.35-second first merge,
then restored before another adapter. This checks mode transitions and the cold
cost guard, not trained-adapter numerical equivalence.

The live comparison runs two optimizer updates per client with one prefetched
rollout. Both measured rollouts may therefore use the initial policy. It tests
real sampling, filtering, gradients, optimizer steps, and publication, but does
not establish updated-policy sampling. The replay experiments separately test
new exact versions. These are distinct claims.

## Limits of the implementation

No policy here is established as uniformly better than KV affinity or the simple
overflow control. The score uses aggregate token rates, not a model of every
continuous-batching or hybrid-attention kernel. Replica observations can be stale
and different frontends can make correlated choices. Local reservations do not
constitute a global semaphore: the authoritative admission gate prevents excess
execution, but thousands of rejected retries remain possible. Fairness applies
to admitted work; it does not guarantee fairness among clients still retrying
outside the sidecar. The revised serving trial's p95 is worse than the simpler
overflow policy despite fewer adapter transfers.

Merging is considered only when the replica has drained. It cannot transform an
in-flight batch or preserve its KV state across the transition. Continuous mixed
demand may therefore never create an amortizable merge opportunity. Engine
concurrency remains fixed at 36: the scheduler chooses which admitted requests to
forward, but does not autotune the engine's concurrency limit. The partial-batch
merge adjustment is a conservative interpolation from a full-batch calibration,
not a measured service curve for every batch composition. Fixed-size
pool tests do not establish Flash autoscaling behavior, failover behavior, or
scaling to hundreds of replicas. The tested native topology is TP1/TP2 with
DP=PP=1. Model/precision-specific numerical validation is still required before
enabling trained-adapter merging; exact unmerge restoration is insufficient.
Replay version turnover makes a saved exact version new to the engine's CPU
adapter registry; it does not guarantee a cold operating-system page cache or
Modal Volume network fetch. Those storage tiers are not isolated by this study.

Artifacts are recorded under `analysis/dlora-20261015`. The earlier eight-H200
attempt with retained pre-update replicas is excluded. Later H200 pool attempts
could not obtain the complete allocation. The replay study uses four TP2 H100
replicas (eight GPUs); live RL uses four TP1 H200 replicas and a four-H100 TP4
trainer. Initial live setup failures from environment JSON escaping and selection
of an incompatible initial checkpoint set are preserved, followed by a corrected
run using the four-GPU checkpoint set. Final tables must identify complete,
failed, and interrupted runs explicitly.
