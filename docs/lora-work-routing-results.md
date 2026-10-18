# Measured-work routing: design and real-GPU results

The concern about additive scores was justified. Converting every term to seconds
did **not** make independent per-request placement work: the first implementation
spread each prefix across 7.59 replicas, versus 1.01 for native KV affinity, and
reduced steady completed-token throughput by 23.9%. Placing sibling samples
together recovered much of that loss. The grouped revision improved the warmed
case's steady throughput by 7.35% and request p95 by 27.36%, but lost 10.06
percentage points of KV hits, increased adapter loads, and had a worse final
drain. In the skewed version-turnover case, the grouped router improved steady
throughput 36.06%, full-run throughput 23.31%, and rollout p95 44.26%, while
increasing adapter movement. It is an experimental policy, not a demonstrated
general replacement for KV affinity. The deployment default remains unchanged.

**The ablation was faster without adapter setup penalties**: +4.88% steady
throughput over the full score, while including the penalties reduced GPU loads
by 5.7%. The cost terms reduced movement but did not earn
a throughput claim. See the exact counts and denominators below.

These are real-GPU **inference replays**, with independent RL-style rollout and
recorded update-delay loops. They do not execute optimizers or measure training
throughput. All timeouts, retries, and failed policy variants remain in the
results. The [Notion report](https://app.notion.com/p/3f71e7f1694981628893fea0f75f2711)
contains the figures; [draft PR #52](https://github.com/modal-projects/spindle/pull/52)
contains the implementation. Earlier native-home routing results are a separate
[historical experiment](lora-kv-training-replay-results.md).

## Policy and request path

Each sampling process owns one `WorkRouter` per inference pool, shared across
all its LoRAs. The path is `sample_task` → local `select_destination` → authenticated
Modal Flash with `Modal-Flash-Upstream` → replica sidecar → SGLang. There is no
central application scheduling service and no routing RPC per physical sample.
The sidecar remains authoritative for the requested immutable adapter version.

For each healthy replica, estimate work in seconds:

```text
backlog = observed remaining prefill / measured prefill rate
        + observed remaining decode / measured aggregate decode rate
        + work in local reservations not yet observed by that replica
        + GPU-load time × distinct pending adapters absent from GPU
        + registration time × distinct pending adapters absent from CPU

setup = GPU-load time if the new adapter is neither resident nor pending
      + registration time if it is neither CPU-registered nor pending

score = backlog + setup
      + placement_group_size × ((prompt_tokens - reusable_tokens) / prefill_rate
                                + output_limit / decode_rate)
```

Choose the lowest score. A numerical tie retains the previous prefix destination;
otherwise version/prefix rendezvous order breaks the tie. These are calibrated
service quantities, not separately tuned affinity/load/eviction weights. The
score estimates drain work rather than exact request completion time. It does
not solve continuous batching, adapter eviction, or cache placement globally.

The adapter terms are additive because the pinned SGLang scheduler handles
registration synchronously, calling its worker and LoRA manager on that path.
GPU adapter loading also runs with overlap loading disabled in this experiment.
Volume reads outside the engine can overlap other work, so charging all exposed
registration as scheduler work is conservative. The first prototype used
`max(backlog, registration)`; source inspection motivated this correction. The
warm comparison has zero registrations, so this correction cannot explain its
improvement. See the pinned [scheduler registration path](https://github.com/modal-projects/sglang/blob/d050d06437d96196fc68d5b4e5c246408790d537/python/sglang/srt/managers/scheduler.py#L4762).

**Placement unit matters.** The first sample of a known generation group chooses
the destination using the whole group's work. Subsequent siblings retain that
choice while it remains healthy. Groups larger than measured decode capacity
are split: capacity 36 makes a 128-sample group become 36/36/36/20. This uses
actual engine capacity, not Flash's target concurrency of 64. Plans are bounded
to 8,192 entries. Retries select independently.

Each physical sample reserves its own predicted work synchronously before
network I/O and releases it on success, failure, or cancellation. Observed
request IDs remove double counting of those local reservations. The entire
unsent group is not atomically reserved, and reservations are not shared across
frontends. The event loop serializes this short CPU operation; no async lock or
global queue is held. Prompt hashes are computed once per sample group.

## Actual cache observations

An opt-in hook in the pinned SGLang scheduler snapshots CPU metadata once per
second. It does not read GPU tensor contents, synchronize CUDA, or run synthetic
cache lookups. A bounded background writer publishes an atomic local file; the
sidecar exposes it through `/routing-state`. Each active frontend coalesces
snapshot polling at most once per second and caches membership for five seconds.
Selection uses local state. Startup or expired observations share one bounded
refresh; observations older than three seconds fall back to native Flash.

Snapshots contain CPU/GPU adapter identities, request IDs, remaining prefill and
decode work, and hashed reusable prefix boundaries scoped to the exact adapter
version. Prefix hashes use 64-token boundaries and exclude the final prompt
token. Snapshots replace the prior set, removing evicted state, and retain at
most 8,192 recent supported boundaries per replica. This is actual cache
inventory, not the old router's recent-prefix history.

The tested Qwen hybrid model uses `UnifiedRadixCache`. A reusable boundary needs
both its full device KV path and a resident Mamba state. Full-attention KV alone
does not suffice. The observer also supports the inspected RadixCache and
MambaRadixCache layouts; unsupported, host-only, salted, and bigram state receives
no credit. It currently supports one DP replica and one pipeline stage per
container. See the pinned [Mamba device-state validator](https://github.com/modal-projects/sglang/blob/d050d06437d96196fc68d5b4e5c246408790d537/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py#L132).

## Configuration and calibration

| Setting | Value |
| --- | --- |
| Pool | `spindle-lora-cache27-work1013a`, us-west, fixed 8 replicas |
| Model / GPUs | Qwen/Qwen3.8-27B, 8 × H200 total, TP=1 |
| Context / memory | 16,384 context limit, static memory fraction 0.8 |
| Adapter rank / targets | Rank 32; q/k/v/o and gate/up/down projections |
| LoRA GPU / CPU capacity | 8 GPU slots, 256 CPU registrations per replica |
| SGLang | `modal-projects/sglang` revision `d050d06437d96196fc68d5b4e5c246408790d537`; LPM; overlap loading disabled |
| Running / queued capacity | Requested 128 running, observed **36**; 4,096 queued per replica |
| Flash | Target concurrency 64; min=max=8, so autoscaling is not tested |
| CPU frontends | 8 actual nonpreemptible containers, 2 vCPU / 8 GiB each, 2 clients each |
| Clients / adapters | 16 independent clients; immutable trained versions 4 and 7, 32 identities total |
| Sampling | Temperature 1; seed `4242 + round*100 + group`; empty stop list → `ignore_eos=True` |
| Prompt / output | 8,192 input = 7,168 reusable prefix + 1,024 round-specific suffix; exactly 512 output |
| Arrivals | 0.5 s client stagger; 300 s new-rollout window; 100-round safety cap |
| Loop | Next rollout overlaps recorded client update/publication delay, approximately 34–96 s; no inter-client round barriers |
| Transport | 240 s attempt timeout; 900 s retry budget; retries included |

Long prompts deterministically sample valid IDs from the saved DAPO vocabulary.
They are synthetic serving prompts, not math-quality evaluations. The source
manifest is `analysis/kv-lora-training-20261013/dapo-clients.json` in the workspace.
Only inference GPUs run in this experiment; trainer fields in the pool recipe do
not imply allocated trainers.

Before measurement, three serialized suffix variants per logical prefix run
through native affinity. The last warmup reports 7,168 cached tokens for every
prefix. All policies use the same eight replica identities and observation hook.
The warmed workload has 8 groups × 8 samples per client, 64 concurrent requests
per rollout, and a maximum of 1,024 outstanding requests across clients. It uses
version 4 throughout, with all 32 identities registered before measurement.

The turnover workload has client 0 at 1 × 128 samples; clients 1–7 at 1 × 64;
clients 8–15 at 8 × 8. Maximum outstanding requests are 1,088, across 72 logical
prefixes. Each client's first two rollouts use version 4 and its third onward
use version 7. All identities are unregistered before setup; only version 4 is
primed. Version 7 therefore requires real CPU registration. Its checkpoint files
already exist on the Volume and may be filesystem-cached: this is not guaranteed
network-cold loading or live optimizer publication.

| Measured service quantity | Warmed profile | Turnover profile |
| --- | ---: | ---: |
| Uncached prefill, tokens/s/replica | 8,828.90 | 8,718.09 |
| Aggregate decode, tokens/s/replica | 1,105.72 | 1,063.78 |
| CPU-resident GPU-cold adapter increment | 45.55 ms | 49.96 ms |
| Exposed CPU registration | 2.435 s | 0.592 s |
| Decode placement capacity | 36 | 36 |

`work_calibrate.py` measures eight replicas concurrently: uncached 8K prefill,
36-request decode, and paired GPU-cold/GPU-hot requests with equal KV hits. Rates
are medians across replicas. The second calibration enlarges the HTTP pool to
1,024 connections, uses cohort engine makespan, and unregisters adapters first.
Compute rates change about 4%; Volume-warm registration is substantially cheaper.
These profiles were not swept or tuned against benchmark outcomes.

The magnitudes matter: under the turnover profile, preserving 7,168 cached
tokens for an eight-sample group saves about **6.58 estimated service seconds**,
versus **0.050 s** for a CPU-resident GPU load and **0.592 s** for an unregistered
adapter. An 85 s predicted backlog can dominate all of those setup costs. The
policy therefore willingly increases adapter replication to relieve pressure;
the ablation checks whether the small setup penalties still help.

A separate serialized greedy control repeated an 8K prompt three times on each
replica. Cache counts were 128, 128, then **8,128/8,192 (99.22%)**, and all 24
16-token outputs had identical hashes. Reuse is possible; hybrid checkpoint
creation requires attention when interpreting the first repeat. This does not
establish bitwise equivalence under arbitrary concurrent batch shapes.
The long-prefix workload changes a 1,024-token suffix each round: the reusable
7,168-token prefix alone accounts for 87.5% of input, and within-group identical
prompt reuse can add more. Cache-hit percentages should be read in that context.

## Warmed long-prefix results

| Metric | Native KV affinity | Individual score, failed | Grouped score |
| --- | ---: | ---: | ---: |
| Completed logical requests | 3,392 | 3,136 | 3,712 |
| Completed tokens/s in [60,300) | 4,994.13 | 3,801.60 | 5,361.07 |
| Full-run output tokens/s | 4,377.78 | 3,721.17 | 3,637.03 |
| Full duration, s | 396.71 | 431.49 | 522.55 |
| Mean request latency, s | 54.55 | 92.80 | 53.71 |
| Request p95, s | 128.79 | 155.71 | 93.54 |
| Mean rollout latency, s | 101.61 | 122.89 | 94.76 |
| Rollout p95, s | 147.42 | 173.20 | 261.05 |
| Token-weighted KV hits | 81.20% | 39.25% | 71.14% |
| First-prefill hot-LoRA hits | 86.56% | 84.00% | 91.50% |
| Hot-LoRA hit / miss counts | 2,937 / 456 | 2,662 / 507 | 3,401 / 316 |
| GPU loads / evictions | 164 / 164 | 287 / 283 | 224 / 224 |
| Summed GPU-load host-call time, s | 9.27 | 16.47 | 11.41 |
| Loads per 1,000 completed requests | 48.35 | 91.52 | 60.34 |
| CPU registrations | 0 | 0 | 0 |
| Extra attempts | 1 | 34 | 5 |
| Mean replicas per prefix | 1.01 | 7.59 | 2.72 |
| Busy observations with an idle replica | 65.15% | 0% | 0% |
| Busy replica outstanding-load CV | 0.653 | 0.134 | 0.186 |
| Replica observations above 8 active adapters, busy | 37.22% | 71.83% | 15.82% |

The individual router recomputed 15.61 million uncached prompt tokens versus
5.22 million for KV affinity, despite completing fewer requests. The cache
inventory initially identified a 7,168-token match for every prefix. Losing
reuse was therefore not simply an absent prefix index. Per-request greedy
placement replicated prefixes widely; queued work overestimation and stale
distributed observations can further encourage spills.

Group placement reduced replication and restored cache reuse, but still consumed
8.78 million uncached prompt tokens. A higher first-prefill LoRA hit fraction
also did **not** imply fewer total adapter loads: batching can improve the
request hit fraction while repeated residency changes still increase. Both
metrics are necessary.

The grouped run's 7.35% steady throughput gain coexists with a 16.92% full-run
throughput loss and a much worse rollout p95. Five 240 s read-timeout retries
extended the drain. Snapshot records for these requests ended much earlier than
the frontend timeout, so engine queueing alone does not explain those waits.
The cause remains unresolved; they are retained, not excluded. The failed
individual run had 34 `RemoteProtocolError` retries clustered late in the run;
KV affinity had one read timeout. These transport effects limit causal claims.

The KV baseline is reused for the grouped comparison, not an independent repeat.
All workload configuration fields match except routing rates gaining the
36-sample placement capacity. The initial individual run also fell back to native
Flash for its first burst; the grouped revision fixes expired-observation refresh
and explicitly routes every attempt. These two changes are not isolated ablations.

## Adapter turnover and skew

| Metric | Native KV affinity | Grouped measured work | Grouped, no adapter cost |
| --- | ---: | ---: | ---: |
| Completed logical requests | 3,456 | 4,224 | 4,288 |
| Completed tokens/s in [60,300) | 4,437.33 | 6,037.33 | 6,331.73 |
| Full-run output tokens/s | 3,698.74 | 4,560.80 | 4,811.79 |
| Full duration, s | 478.40 | 474.19 | 456.27 |
| Mean request latency, s | 74.87 | 64.60 | 58.34 |
| Request p95, s | 172.46 | 100.49 | 89.32 |
| Mean rollout latency, s | 116.44 | 84.53 | 86.79 |
| Rollout p95, s | 206.88 | 115.32 | 141.16 |
| KV token hits | 85.40% | 81.60% | 81.95% |
| Hot-LoRA hits | 98.35% | 96.88% | 96.55% |
| Hot-LoRA hit / miss counts | 3,400 / 57 | 4,093 / 132 | 4,143 / 148 |
| GPU loads / evictions | 56 / 56 | 132 / 132 | 140 / 140 |
| Loads per 1,000 completed requests | 16.20 | 31.25 | 32.65 |
| Summed GPU-load host-call time, s | 2.80 | 6.80 | 7.23 |
| CPU registrations | 34 | 69 | 75 |
| Extra attempts | 1 | 1 | 3 |
| Busy observations with an idle replica | 79.76% | 0.65% | 0.00% |
| Busy replica outstanding-load CV | 1.071 | 0.218 | 0.204 |
| Mean replicas per exact-version prefix | 1.01 | 1.94 | 1.97 |
| Summed registration HTTP-call time, s | 379.36 | 251.93 | 99.19 |
| Version-7 requests / clients reaching v7 | 1,280 / 14 | 2,048 / 16 | 2,112 / 14 |
| Registrations per 1,000 version-7 requests | 26.56 | 33.69 | 35.51 |

The grouped router improved steady throughput **36.06%**, full-run throughput
**23.31%**, mean request latency **13.72%**, request p95 **41.73%**, mean rollout
latency **27.41%**, and rollout p95 **44.26%**. The main observed mechanism is
less concentrated load, at the expense of cache locality and adapter movement.
GPU loads increased **135.71%** in total and **92.86%** per completed request;
CPU registration calls increased **102.94%**. This is not a LoRA-thrashing win.
The baseline's outstanding adapter working sets never exceeded eight in the
busy observations: this case primarily exercises affinity overload and version
arrival; the warmed ordinary-group case creates more adapter working-set pressure.

Registration HTTP-call time includes waiting for engine handling, not just
reading or copying weights. More registration calls taking less summed time
does not demonstrate faster transfers. The router's idle calibration estimates
setup service; the separately estimated backlog is intended to represent waiting.

Independent progress is directly visible: grouped client 4 began rollout 1 at
24.29 s, while client 14's rollout 0 continued until 122.47 s. Across the 50
common client/round pairs, 38 completed faster, and 14 of 16 clients had lower
mean common-rollout latency. Clients 1 and 6 became slower. These descriptive
comparisons retain retries and do not remove interference from different
closed-loop arrival times.

Both policies had one 240 s read timeout. The baseline failed attempt was
observed at the backend around 6.39–44.00 s but timed out at 245.81 s; the grouped
attempt was observed around 222.99–341.37 s but timed out at 461.90 s. Both stalled
while awaiting HTTP response headers. The missing interval is not explained by
observed engine queueing; the transport cause remains unresolved. One grouped
replica-discovery timeout recovered, with 100% of generation attempts still
explicitly routed. No final request failed.

**The adapter-cost ablation does not establish a benefit from those terms.**
Removing both setup terms increased steady throughput from 6,037.33 to 6,331.73
output tokens/s (**+4.88%**) and full-run throughput from 4,560.80 to 4,811.79
(**+5.50%**). Conversely, including them reduced GPU loads 140→132 (**−5.71%**)
and CPU registrations 75→69 (**−8.00%**), but worsened request mean and p95 by
10.73% and 12.51%. Rollout mean and p95 favored including the terms by 2.61%
and 18.31%; the ablation had three read-timeout retries versus one and two clients
never reached version 7 before admission closed.

Both grouped variants used the same workload, profile, replicas, and trace hash.
They ran sequentially, with one trial each and different realized arrivals.
These results support the grouping/load-balance mechanism and a small reduction
in adapter movement from setup penalties. They do **not** establish that those
penalties improve request or training throughput, or that removing them always
wins. No default was changed in response to this single ablation.

## Interpretation and limits

The experiment directly rejects the assumption that adding calibrated terms is
enough. Joint group placement matters, and the current backlog estimate remains
imperfect. Queued requests can advertise full prefill before their cache match is
resolved; sidecar-pending requests also count their full prompt. The score counts
one load per pending adapter and cannot predict repeated reloads when more than
eight interleave. It lacks a future cache-eviction cost. All of these can move
requests even when most of their prefix would be reused at the original host.
Prefill cost is also not universally linear in uncached tokens: suffix length,
existing context, and batching affect service rate. These profiles cover one
tested context/concurrency regime, not a complete kernel cost curve.

For now, retain this as an opt-in research policy. Do not change the production
default or claim training throughput or autoscaling improvements from these
results. Additional checks would need repeated randomized policy order, actual
optimizer loops, variable early-EOS lengths, calibrated heterogeneous replicas,
and replica scale-up/down. Each client stays on one frontend here; same-LoRA
fan-in across many frontends is not tested. This trial supplies neither
confidence intervals nor statistical significance.

Steady throughput counts completed output tokens in the fixed [60,300) interval;
it is not instantaneous GPU decode rate. Full-run throughput includes launch,
all requests and retries, and final recorded update delays. Faster policies can
start more rollouts before the cutoff, so total completed work differs. KV ratios
use successful logical requests; hot-LoRA counters include all physical engine
admissions, including duplicates. Adapter load host time is bookkeeping wall
time summed across replicas, not GPU critical-path savings.

Every completed arm checks exact requested versions, fixed output lengths,
finite aligned log probabilities, route identity, native admission accounting,
eight distinct CPU workers, stable replica membership, and zero final engine
requests. Unit/integration regression coverage: 174 tests passed. Grouped
selection CPU time was p50/p95 **38/212 microseconds** in the warmed case and
**55/253 microseconds** during turnover. These exclude hashing and refresh waits,
which remain included in end-to-end latency. Engine snapshot CPU time, once per
second, was p95 **1.86 ms** and **1.73 ms**, respectively. Snapshot JSON was p95
approximately **9.0 KB** and **9.4 KB** per replica, excluding HTTP framing and
other routing-state fields. This is measured at eight replicas, not a scalability
claim for hundreds. The final resource audit confirms **peak 8 H200s, zero active GPUs**, and no
cleanup error. All three turnover arms drained successfully. The final small
reservation cleanup removes empty per-host maps; it does not change placement
and passed the 10 relevant tests after the 174-test regression suite.

## Reproduction and evidence

The [machine-readable metrics](lora-work-routing-metrics.json) include all six
measured arms, their full configuration dictionaries, per-version counts,
per-client rollout counts, service profiles, and the final resource audit.

Raw records, configurations, calibration, monitor logs, summaries, and PNG/PDF
plots are under `analysis/work-routing-20261013` in the workspace. The measured
individual implementation is commit `6be8811`; the grouped implementation is
`283d689`. The final reporting/calibration changes do not alter the grouped
routing policy. See [benchmark instructions](../scripts/benchmarks/lora_routing/CACHE_BENCHMARK.md)
for commands and metric definitions. Preserve the raw retry records when
recomputing summaries.
