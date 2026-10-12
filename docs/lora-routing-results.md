# LoRA routing benchmark — 2026-10-11

The final router delivered **24.4% higher inference throughput** and **16.5% lower
p95 request-group latency** in a matched replay of trained adapters. All four
waves improved throughput. The independent-training comparison also observed
27.7% lower mean rollout latency, but output tokens per experiment wall-clock
second were 2.0% lower. That denominator includes unequal initialization of later
clients: their first-rollout spread was 485 s with affinity versus 139 s with
naive routing. This is not a measured steady-state training regression or gain.

## Matched inference replay

Both arms replayed the same 16 adapters from the completed naive training run,
with identical immutable snapshot manifests, prompts, seeds, and per-wave
submissions. Each arm used eight H200 replicas and eight independent router
caches in a four-CPU replay worker. All eight replicas had to pass readiness,
then each adapter received a short warmup before timing began. Four waves each
submitted 1,024 generations capped at 512 tokens. Snapshot versions alternated
4, 7, 4, 7: the first two post-training publications in this run.

| Metric | Naive | Affinity | Change |
| --- | ---: | ---: | ---: |
| Output tokens | 2,096,733 | 2,097,143 | +0.02% |
| Total measured wave time | 328.6 s | 264.1 s | -19.6% |
| Output tokens/s | 6,381 | 7,940 | **+24.4%** |
| Mean request-group latency | 44.34 s | 42.40 s | -4.4% |
| p95 request-group latency | 76.17 s | 63.63 s | **-16.5%** |

Throughput is total generated tokens divided by the sum of all measured wave
durations. Group latency covers all eight generations in a request; p95 is
computed over all 512 groups per arm. Startup, warmup, and training are excluded.
Discovery refreshes and adapter registrations during measured requests remain
included. The wave barrier belongs to this inference-only replay and is not used
in the independent training benchmark.

| Wave / snapshot | Naive tokens/s | Affinity tokens/s | Gain |
| --- | ---: | ---: | ---: |
| 0 / 4 | 6,693 | 7,663 | +14.5% |
| 1 / 7 | 5,411 | 8,297 | +53.3% |
| 2 / 4 | 6,822 | 7,210 | +5.7% |
| 3 / 7 | 6,843 | 8,766 | +28.1% |

This is one paired experiment with four waves, not a confidence interval or a
claim across all workloads. The varying per-wave gains are retained rather than
selecting the best wave. Both arms completed every group with exact-version
checks and finite, aligned output logprobs. The final router revision was
`25192e7`. The workload hash is identical across arms.

- [Replay summary and hashes](../scripts/benchmarks/lora_routing/results/replay.json)
- [Every measured request group](../scripts/benchmarks/lora_routing/results/replay-groups.csv)

`report_replay.py RAW_REPLAY.json --output SUMMARY.json --groups-csv GROUPS.csv`
recomputes these results from `replay.py` output.

## Observed changes in routing behavior

Additional analysis of the saved replay observations shows fewer copies of each
new adapter version across the pool. The same 16 version-7 policies were sampled
in waves 1 and 3, totaling 2,048 generations per arm. They required 111 distinct
replica/adapter registrations with naive routing versus 38 with affinity routing:
**65.8% fewer CPU registrations**, or 6.94 versus 2.38 replica copies per policy.
This measures CPU adapter registration, not GPU onloads or unloads.

The observer joined during wave zero, so it cannot separate that wave's loading
cost from warmup. Counters were stable before the first version-7 registrations,
and observation continued until both pools were idle after wave three. Across
the remaining three waves, including new placements of version 4 on its return:

| Metric, waves 1–3 | Naive | Affinity | Change |
| --- | ---: | ---: | ---: |
| CPU adapter registrations | 140 | 51 | -63.6% |
| Summed registration RPC duration | 442.38 s | 244.32 s | -44.8% |

The duration counter wraps `/load_lora_adapter`: it includes SGLang-side waiting
and loading, excludes preceding snapshot resolution and the sidecar's adapter
lock wait, and is summed across eight replicas. It is neither GPU transfer time
nor elapsed experiment time. Fewer calls reduced their summed duration, although
mean duration per call increased from 3.16 s to 4.79 s. The data does not show
that an individual load became faster.

For these measurements, all eight replica identities remained unchanged and
counters were monotonic. Before the new-version phase, all registrations were
accounted for by active/recent version-4 identities. At the final idle observation,
every replica's recent identity count equaled its lifetime registration counter,
so the 32-entry recent history was complete. The report validates these conditions
before reconstructing version-7 placements. Boundary observations and the source
observation-file hash are retained in `results/replay.json`. Reproduce with:

```bash
python scripts/benchmarks/lora_routing/report_replay.py RAW_REPLAY.json \
  --observations REPLAY_OBSERVATIONS.jsonl --router-commit 25192e7 \
  --output SUMMARY.json --groups-csv GROUPS.csv
```

The latency distribution also shows that the benefit concentrates in the slow
tail rather than uniformly accelerating every request group:

| Eight-sample group latency, all four waves | Naive | Affinity | Change |
| --- | ---: | ---: | ---: |
| Median | 46.68 s | 45.76 s | -2.0% |
| p75 | 53.34 s | 59.23 s | +11.0% |
| p95 | 76.17 s | 63.63 s | -16.5% |
| p99 | 85.98 s | 67.23 s | -21.8% |
| Maximum | 96.77 s | 72.71 s | -24.9% |
| Groups taking more than 75 seconds | 28 / 512 | 0 / 512 | — |

The p75 regression is retained: this is a placement tradeoff, not a uniform
latency improvement. The shorter slow tail is consistent with the shorter time
needed to drain each replay wave. Neither registration counters nor these latency
distributions isolate GPU eviction/H2D costs, prefix-cache effects, or scheduling
effects. GPU onload/unload counts and transfer durations were not instrumented.

## Workload and resources

Both arms used Qwen/Qwen3.8-27B, 16 distinct rank-32 trainable adapters, and four
DAPO math PPO updates per adapter. Each update sampled 16 groups of eight
generations, capped at 1,536 tokens: 8,192 measured sequences and approximately
12 million output tokens per arm. The generation submission ceiling was 2,048
concurrent sequences. Each independent client prefetched one rollout while its
own training progressed; policy lag was bounded to one update. Starts were
staggered by five seconds without cross-client step barriers.

Each arm had eight single-H200 inference replicas and one eight-H100 trainer
(TP=2). Inference ran in `us-west`; trainers used unrestricted region placement
to avoid capacity queues. Requested SGLang limits were 128 running requests and
eight adapters per batch. The model's hybrid/Mamba capacity actually limited
running requests to 36 per replica in the logs. Sidecar active-request counts
include queued work and must not be read as simultaneous GPU sequences.

The frozen dataset rows, tokenizer revision, prompt suffix, sampling seeds and
parameters match across arms; the three workload hashes are recorded in the
machine-readable results. Measured intervals start at the first training rollout,
after the first client's initialization, and end after the last
client's final updated sampling policy is ready. Per-client step timing includes
rollout, training queueing, optimizer work, and publication. It may overlap other
steps because clients prefetch independently.

Later clients initialize while earlier clients are already training. Their
creation, initial publication, and warmup therefore remain inside the global
measurement window. Earlier descriptions incorrectly said client initialization
was excluded without this qualification.

Pools were deployed in advance, but the training harness did not gate on all
eight replicas becoming healthy. The baseline's first eight-replica observation
was 6.4 seconds after measurement began; affinity already had eight. This is
another reason to keep the training result separate from the fully warmed replay.

## Completed training comparison

| Metric | Naive | Affinity | Change |
| --- | ---: | ---: | ---: |
| Completed optimizer updates | 64 | 64 | — |
| Generated output tokens | 11,972,813 | 11,984,417 | +0.1% |
| Measured elapsed time | 1,361.4 s | 1,390.6 s | +2.1% |
| Output tokens/s including training and later-client initialization | 8,794 | 8,618 | -2.0% |
| First-to-last client rollout start | 139.4 s | 485.4 s | +346.0 s |
| Mean per-client four-update duration, from its own first rollout | 1,138.5 s | 907.5 s | -20.3% |
| Mean rollout batch latency | 274.9 s | 198.8 s | -27.7% |
| Median rollout batch latency | 274.7 s | 195.8 s | -28.7% |
| p95 rollout batch latency | 462.8 s | 345.0 s | -25.5% |
| p95 sampling group latency | 323.6 s | 240.3 s | -25.7% |
| Mean rollout-to-policy-ready step | 332.5 s | 323.0 s | -2.9% |
| p95 rollout-to-policy-ready step | 541.7 s | 507.6 s | -6.3% |
| Observed adapter registrations | 362 | 292 | -19.3% |
| Mean active adapters per busy replica | 6.37 | 3.96 | -37.9% |
| Busy replica time with >8 active adapters | 17.56% | 0.16% | — |

Registration counts measure CPU `/load_lora_adapter` calls, not GPU transfers or
evictions. Observations were sampled about every five seconds; working-set
statistics are time-weighted estimates, and counter boundaries have the same
sampling granularity (the baseline boundary includes its first warmup adapter
registration). Peak active adapters on one replica fell
from 12 to nine. These observations are consistent with better locality but do
not establish a GPU transfer speedup.

Both arms completed all updates with finite metrics and successful optimizer
status. All 16 clients in each arm had nonzero gradients and sampled policies
after an optimizer update. Naive had 81 mixed-reward groups and 44 nonzero-gradient
updates; affinity had 65 and 43. Output logprobs were finite and aligned to tokens;
exact adapter/version response checks remained enabled. These short runs validate
real learning operations, not convergence or identical model quality.

The first rollout finished after 94.8 s with naive routing and 47.6 s with
affinity. Client launches continued until about 79 s in both arms. In the affinity
run, training thus began before the later clients had initialized. The first
backend forward/backward took 117.8 s versus 82.6 s in the naive run. Model
acceptance and forward/backward execute in the same
[engine work loop](../src/spindle/engine/server.py#L438), so a running training
operation delays subsequent model acceptance. Several later affinity clients
waited 91–120 s for creation and 91–239 s for initial publication. Initial
publication took approximately 4–6 s per client in the naive run.

These timestamps and the execution path explain an initialization/queueing
interaction; they do not prove steady-state trainer compute became slower.
Mean client-observed training spans were 47.6 s (naive) and 99.2 s (affinity).
Publication spans **after optimizer updates**, excluding the initial publication,
were 10.0 s and 20.2 s. The previously reported 9.0 s and 32.9 s included initial
publication and should not be read as post-update publication costs. These spans
include queueing. Consequently,
different training timing also changed inference arrivals. Treat the latency
comparison as an observed system result, not a controlled causal estimate or a
claim of improved end-to-end training throughput.

Per-client four-update duration was 20.3% lower on average, but clients also
experienced different concurrency; that number is not a replacement throughput
claim. A controlled training comparison must initialize clients and warm the
trainer before measurement, then retain the same staggered, independent training
loops without cross-client step barriers. Such a training rerun has not yet been
performed.

The summaries retain every client's rollout, step, training, publication, reward,
gradient, and policy-version measurements:

- [Naive training results](../scripts/benchmarks/lora_routing/results/naive-training.json)
- [Affinity training results](../scripts/benchmarks/lora_routing/results/affinity-training.json)

## Correctness and iteration

The routing, sampling, sidecar, provider, and configuration checks passed 211
tests locally. After the idle-burst fix, the affected routing/sampling/sidecar
suite passed 95 tests, including the new idle-burst regression. One native-SGLang
module was skipped locally because SGLang was
unavailable; all 32 of its tests passed inside the actual rollout image on CPU.
A live two-replica Flash probe delivered 40/40 concurrent requests to their exact
selected replica, and a stale upstream returned 503.
The CPU image check is reproducible with `check_sglang.py`; the live routing check
uses `probe_flash.py` in the benchmark directory.

Early pilots with 1,024-token truncation produced zero gradients and are excluded
from performance claims. An initial full baseline was interrupted by a cleanup
503 after one client had completed; the harness now records cleanup warnings
without discarding completed work. An initial affinity run exposed load
under-counting when new local arrivals were hidden behind an existing replica
load; the router now adds arrivals since observation, with a regression test.
Only the completed reruns above enter the comparison.

The independent training comparison used the placement algorithm from commit
`59d00d1`. A later review of the fixed-wave replay found that stale observations
could send an entire post-idle burst through naive fallback. The final router
coalesces those requests behind one three-second-bounded refresh, then falls back
if observations remain stale. Fresh requests remain nonblocking. The supplemental
replay tests this final behavior; the training numbers above precede that fix.

The first replay attempt correctly rejected guessed, nonexistent snapshot
versions. The harness now discovers and checks immutable manifests before running.
A second attempt stopped when one pool missed the five-minute readiness limit;
the limit is now ten minutes, with a thirty-minute function deadline. Neither
attempt contributes performance data. The final replay reused initialized GPU
pools when refreshing CPU workers for the idle-burst fix.

A CPU microbenchmark of 100,000 selection/release pairs measured 15.21 microseconds
per request with eight replicas and 52.38 microseconds with 32 replicas. This
measures local selection/reservation, excluding asynchronous observation traffic,
HTTP, tokenization, and generation. Run `benchmark_router.py` to reproduce it.

Task-level resource auditing, including interrupted trials and replay attempts,
recorded a peak of **32 allocated GPUs**, below the requested 48-GPU limit. All
benchmark GPUs were released at completion. The final
[resource audit](../scripts/benchmarks/lora_routing/results/resource-audit.json)
records zero active GPUs.

See [the routing design and reproduction guide](lora-routing.md) for deployment,
independent training clients, observation collection, and report commands.
