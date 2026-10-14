# Pool routing research, October 14

This plan was written before collecting the new experiment results. The previous
October 13 runs are exploratory evidence, not independent replications.

## Questions and hypotheses

1. Does controlling adapter replication reduce cache fragmentation without
   trapping hot adapters on a small set of replicas?
2. Does the benefit survive long reusable prefixes, changing adapter versions,
   and dependent multi-turn generation?
3. Are improvements explained by adapter traffic, prefix computation, or load
   balance? A higher cache hit rate alone is not success.
4. Does a decentralized implementation preserve correctness under stale state
   and concurrent frontends, with low request-path CPU overhead?

The first candidate restricts cold placement to a deterministic adapter domain
of ceil(outstanding adapter demand / engine running capacity) replicas. Model
identity keeps the ordering stable across versions; cache credit is strictly
version-specific. Existing prefix matches outside the domain remain eligible.
A domain choice cannot exceed the globally best predicted work by more than
one engine-sized batch of the incoming request's service work. This is a
heuristic bound on an estimate, not a theorem about actual latency or a global
admission guarantee. Inaccurate demand snapshots and domain collisions are
explicit failure cases to test.

## Baselines and workload families

Use native Flash KV affinity as the main baseline, previous grouped measured-work
routing as a strong baseline, and default Flash as a context baseline. Include
the candidate's unrestricted-placement ablation (the measured-work baseline).
Published systems are compared architecturally; do not label a partial routing
adaptation as a full dLoRA, Preble, or LMetric implementation.

Use the same pool and exact saved trained adapter identities for all policies.
Reset GPU adapters and KV state between arms, with identical explicit warmup for
warmed cases. Freeze workload seeds; reverse or randomize policy order in repeats.
Pilot runs and policy-selection runs are reported separately from final repeats.

- General serving: 32 adapter identities, independent streams, 4–8K prompts,
  128–512 output tokens; enough simultaneous requests to exceed engine capacity.
- Single-turn RL replay: 16 independent clients, groups of 8–128 generations,
  repeated bursts with staggered starts and recorded per-client update delays.
  Both warmed prefixes and independently arriving saved adapter versions.
- Multi-turn: dependent prompts append the actual preceding response and a
  controlled tool-result segment. Exact token lengths and dependency timestamps
  are saved. This models serving dependencies; it does not measure reward or
  live optimization throughput.

Measure full finite-work makespan, completed output tokens/s, mean/p50/p95/p99
request latency, per-client rollout/trajectory latency, KV token hit ratio,
first-prefill hot-adapter hits, GPU loads/evictions, registration count, observed
queue imbalance, working-set sizes, retries, and router/observer overhead.
Preserve errors and retry time in end-to-end metrics. Separate steady-window
rates from full-run rates, and avoid request-level confidence intervals that
treat correlated requests as independent trials.

## Compute and evidence

The first allocation is eight H200s, with an independent 180-minute shutdown
monitor (24 GPU-hours maximum for this allocation). Peak allocation remains
below the user's 48-GPU ceiling. A separate calibration checks the deployed
engine's real running capacity and exact-version cache correctness. Final
resource accounting must show zero active experiment GPUs.

Raw evidence lives in `analysis/pool-paper-20261014` outside the repository.
The paper must distinguish new measurements, previous exploratory results,
analytic examples, and untested hypotheses. Positive results are not guaranteed.

## Protocol amendment, October 14 at 02:10 UTC

The initial four-arm turnover experiment completed without retries. The warmed
HTTP/1 comparison then produced repeated 360-second read errors. At least one
request disappeared from replica observations about 293 seconds before the
frontend failed. Increasing the client timeout did not address that behavior.
The suite was stopped after saving the completed KV+preparation arm; its logs,
raw data, and interruption audit remain in the artifact directory.

A read-only proxy probe established HTTP/2 support with 1,024 concurrent streams.
The replacement suite validates actual HTTP/2 generation and cache equivalence
first, then compares both policies using that transport. This is a separate
experimental phase; transport changes are not credited to routing. The core
HTTP/2 baseline is KV affinity with identical cohort preparation. Warmed trials
use three rounds so reversed-order repetitions fit the allocation; turnover
retains four rounds and dialogue retains three. The exact commands and order
are recorded by `paper_suite.py --phase http2` before each run. The follow-up
includes 64 adapters with 512 concurrent streams and 512-token outputs, and a
300-second longer-decode RL arrival window if the bounded budget permits.

The independent monitor was handed off without a gap to an absolute deadline
45 minutes later, preserving the eight-GPU ceiling. The revised maximum is
225 minutes (30 GPU-hours), still below the user's 48-GPU peak limit. No
additional inference pool or training GPUs are allocated. The handoff and
eventual zero-GPU audit are preserved with the results.
