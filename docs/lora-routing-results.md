# LoRA routing benchmark — 2026-10-11

The completed independent-training comparison reduced mean rollout latency by
27.7% and p95 rollout latency by 25.5%. End-to-end training throughput did not
improve: it was 2.0% lower. Training/publication queueing differed between arms,
so the training experiment alone does not isolate the router's contribution.

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
machine-readable results. Measured intervals exclude deployment and client
initialization, start at the first training rollout, and end after the last
client's final updated sampling policy is ready. Per-client step timing includes
rollout, training queueing, optimizer work, and publication. It may overlap other
steps because clients prefetch independently.

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
| Output tokens/s including training | 8,794 | 8,618 | -2.0% |
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

Mean client-observed training spans were 47.6 s (naive) and 99.2 s (affinity);
publication spans were 9.0 s and 32.9 s. These spans include queueing. Consequently,
different training timing also changed inference arrivals. Treat the latency
comparison as an observed system result, not a controlled causal estimate or a
claim of improved end-to-end training throughput.

The summaries retain every client's rollout, step, training, publication, reward,
gradient, and policy-version measurements:

- [Naive training results](../scripts/benchmarks/lora_routing/results/naive-training.json)
- [Affinity training results](../scripts/benchmarks/lora_routing/results/affinity-training.json)

## Correctness and iteration

The routing, sampling, sidecar, provider, and configuration checks passed 211
tests locally. One native-SGLang module was skipped locally because SGLang was
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

A CPU microbenchmark of 100,000 acquire/release pairs measured 15.21 microseconds
per request with eight replicas and 52.38 microseconds with 32 replicas. This
measures local selection/reservation, excluding asynchronous observation traffic,
HTTP, tokenization, and generation. Run `benchmark_router.py` to reproduce it.

See [the routing design and reproduction guide](lora-routing.md) for deployment,
independent training clients, observation collection, and report commands.
