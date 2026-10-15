# Model placement with overflow

This PR retains `model-spill`, the simple model-placement policy evaluated in
the serving study. It improves throughput over the matched Flash KV-affinity
baseline on the five comparisons below. It does not establish a universal
speedup: the long-decode result is nearly flat, with slightly worse p95 latency.
There is no completed live-RL comparison for this policy.

Enable it in a deployment recipe:

```python
inference_lora_routing = "model-spill"
```

The default remains `"affinity"`. The sampling API is unchanged. The option
applies to exact-version LoRA requests; base and latest-version requests use
the existing path. The native catalog observer supports TP engines with DP=PP=1;
other configurations fall back to Flash affinity.

## Retained algorithm

Each `execute_sample` container has one router per inference pool, shared by
all models. No coordinator or distributed lock is on the request path.

1. Discover Flash replicas every five seconds and poll their sidecars in
   parallel, at most once per second per active router. The native scheduler
   publishes only its CPU adapter catalog, through a bounded background writer.
   Sidecars expose active requests, their IDs and adapter identities.
2. Assign preferred replicas to model identities, excluding the version suffix.
   Sort the observed model catalog and assign each model to a replica with the
   fewest assigned models; rendezvous hashing breaks ties. Retain catalog and
   topology entries for ten seconds to tolerate transient missing observations.
3. Estimate load as sidecar active requests plus this frontend's reservations
   whose request IDs are absent from that observation. Observations older than
   three seconds are ineligible.
4. Divide sibling samples into chunks no larger than the engine's configured
   `max_running_requests` (or the deployment target concurrency if unspecified).
   For a new chunk, retain the preferred replica when its load is at most
   `1.25 * mean(eligible_replica_loads) + chunk_size`. Otherwise choose the least
   loaded eligible replica. Exact-version affinity hashing breaks load ties;
   the previous affinity destination wins an equal-load tie.
5. Keep subsequent siblings in the chunk on its selected replica while eligible.
   A retry makes a fresh single-request choice. Send through the authenticated
   Flash gateway with `Modal-Flash-Upstream`, removing conflicting affinity
   headers. Release the local reservation on completion, error or cancellation.
   Transport failures, unhealthy responses and wrong-version results trigger a
   short local cooldown. Sampling still validates the returned weight version.

`select_destination` is synchronous and contains no awaits or locks. Fresh
`refresh` calls return immediately while one background poll proceeds. Initial
or stale callers share a discovery operation bounded to three seconds. If
there is no usable observation, requests use Flash's existing affinity route.

The overflow coefficient is the tested heuristic, not a fitted latency model.
This policy promotes reuse through placement and sibling co-location; it does
not query native KV-prefix matches, migrate KV caches, reserve global capacity,
merge adapters, or alter the engine's concurrency and admission settings.

## Historical GPU results

Throughput is completed output tokens divided by the wall-clock generation
makespan, excluding the collector's trailing observation interval. Latencies
include request retries and waiting. Each row compares the same workload and
engine configuration; both arms had merging disabled. Values below are
**KV affinity → model placement with overflow**.

| Workload | Output tokens/s | Throughput gain | Mean request time, s | p95 request time, s |
| --- | ---: | ---: | ---: | ---: |
| 64 versions, trial 1 | 1,629.3 → 2,680.7 | +64.5% | 120.2 → 73.4 | 310.0 → 175.2 |
| 64 versions, trial 2 | 1,640.3 → 2,571.8 | +56.8% | 121.8 → 76.4 | 295.7 → 184.2 |
| Long-prefix independent bursts | 4,053.7 → 5,307.4 | +30.9% | 133.8 → 98.0 | 256.7 → 195.6 |
| Multi-turn | 4,493.4 → 5,027.0 | +11.9% | 36.8 → 34.8 | 74.4 → 66.0 |
| Short-prompt, long decode | 6,081.3 → 6,151.1 | +1.1% | 169.9 → 166.3 | 322.5 → 326.8 |

| Workload | KV token hit ratio | Hot-LoRA request hit ratio | GPU LoRA loads | Mean executed decode batch |
| --- | ---: | ---: | ---: | ---: |
| 64 versions, trial 1 | 15.0% → 49.5% | 36.5% → 60.8% | 1,295 → 657 | 11.5 → 20.4 |
| 64 versions, trial 2 | 14.7% → 47.5% | 36.1% → 60.3% | 1,301 → 677 | 11.7 → 19.3 |
| Long-prefix bursts | 62.0% → 78.7% | 76.9% → 96.9% | 570 → 94 | 26.3 → 34.8 |
| Multi-turn | 75.5% → 81.1% | 84.6% → 95.8% | 249 → 82 | 29.1 → 32.3 |
| Long decode | 50.7% → 56.3% | 84.5% → 90.1% | 77 → 31 | 29.9 → 33.0 |

KV hit ratio counts cached prompt tokens divided by input prompt tokens. Hot-LoRA
hits count requests whose adapter was already GPU-resident when admitted.
Loads count GPU adapter loads, including reloads; they are not network transfers.
These counters support reduced thrashing and larger useful batches as mechanisms;
they do not isolate how much time each mechanism saved.

### Configuration

All comparisons used Qwen3.8-27B, BF16, rank-32/alpha-32 adapters, four TP2 H100
replicas (eight GPUs), eight CPU cores and 128 GiB RAM per replica, a 16,384-token
context, 36 running requests and eight GPU adapter slots per replica, and a
256-entry CPU adapter registration capacity. The historical harness used the
same 36-running-plus-four-waiting sidecar admission gate and a native queue limit
of 40 in both arms. Requests used HTTP/1; forced generation lengths used
`ignore_eos=True`. The long-decode row uses the corrected uncapped HTTP transport.

| Workload | Demand and concurrency | Requests | Input / output tokens per request |
| --- | --- | ---: | --- |
| 64 versions | 16 models × 4 versions; 512 streams, 4 requests each; 8 local router instances in one CPU driver | 2,048 | 8,192 input, 7,168 shared-prefix; 512 output |
| Long-prefix bursts | 16 independent clients, 8 CPU frontends, 3 rounds; 1 hot client × 128 samples, 7 × 64, 8 × (8 groups × 8); maximum 1,088 logical samples | 3,264 | 8,192 input, 7,168 shared-prefix; 1,024 output |
| Multi-turn | 16 clients, 8 CPU frontends, 2 groups × 8 samples, 3 rounds × 3 turns; maximum 256 logical samples | 2,304 | 8,192 → 9,472 → 10,752 input; 1,024 output per turn and 256 synthetic tool tokens between turns |
| Long decode | 12 trained adapters, 8 CPU frontends, 2 rounds; 11 clients × (4 groups × 8), hot client × 128; maximum 480 logical samples | 960 | Approximately 200-token math prompts; 4,096 output |

Long-prefix and multi-turn clients use version 4 for their first two rounds and
version 7 for the third. Clients start at 0.5-second offsets and advance
independently. Recorded training/update delays are replayed between rollout
rounds; these are serving replays, not measured live training speedups. Multi-turn
requests append actual generated continuations. Long-decode requests use trained
version-4 adapters; its CPU frontends were in us-east and GPUs in us-west in both
arms. Workload sizes correspond to 1.05M, 3.34M, 2.36M and 3.93M output tokens,
respectively.

## Scope and limitations

- The GPU trials used SGLang commit
  `708d38f143697e25c3e687c20fc0b043930c8416` for both arms. This extracted PR keeps
  current main's SGLang v0.5.21 and native admission path. It has not been
  GPU-benchmarked on that newer integration. Historical gains are evidence for
  the policy, not measurements of this exact new build.
- The control supplied explicit affinity keys. Current main also derives
  conversation affinity automatically; that newer behavior was not separately
  benchmarked. The control represents Flash KV-affinity routing under matched
  study settings, not every production deployment of Spindle's default.
- Only the 64-version comparison has two repeated trials here. There are no
  confidence intervals establishing a meaningful 1.1% long-decode win. Its p95
  latency worsened 1.3%. This policy also did not beat every other experimental
  policy on every workload.
- Placement balances model counts, not model demand. Overflow handles observed
  pressure, but a chunk remains sticky and simultaneous frontends can act on stale
  observations. A catalog change can move preferred destinations. Autoscaling
  under these conditions has not been validated by these fixed-pool trials.
- The historical admission gate generated many 503 retries; multi-turn had
  24,453 → 28,324 and long decode 112,942 → 113,569. Higher throughput does not
  imply fewer rejected attempts. Neither arm had extra engine admissions in
  these comparisons. The long-prefix KV arm had one transport error.

The extraction matched the archived implementation on 30,000 deterministic
randomized routing decisions, including stale views, local reservations, group
chunks and replica failures. CPU tests cover routing, concurrent refresh,
cancellation, exact-version retries, Flash header handling, discovery and the
native observation hook. These checks do not replace GPU validation.

Exact aggregate values are in [model-placement-results.json](model-placement-results.json).
The prior implementations, experiment drivers and full research report remain on
the [archived study branch](https://github.com/modal-projects/spindle/tree/codex/dlora-serving-study-20261015).
The alternative experimental policies and merge/unmerge machinery are excluded
from this PR's diff against main.
