# LoRA and KV cache comparisons

The measured results and plots are in [the Notion report](https://app.notion.com/p/3f71e7f1694981628893fea0f75f2711?pvs=204)
and [the current KV baseline comparison](../../../docs/lora-kv-training-replay-results.md).
The [original three-policy comparison](../../../docs/lora-cache-routing-results.md)
describes an earlier router.
The [measured-work experiment](../../../docs/lora-work-routing-results.md) adds
engine-backed observations, grouped placement, and a service-cost ablation. It
uses timed independent replay loops; the older fixed-work fixture remains
available below.

This benchmark compares the existing default Flash path, the current LoRA-aware
router, and native Flash affinity using a stable conversation key. It does not
implement LMETRIC. The workload runs on real inference GPUs with frozen adapters;
choose either synthetic repeated prefixes or independent DAPO client replay.
Historical results describe the policy at their recorded commit, not necessarily
the current router.

## Workload

The default configuration uses Qwen/Qwen3.8-27B on eight H200 replicas (TP=1),
rank-32 LoRA, eight GPU adapter slots, 256 CPU adapter registrations, and a fixed
pool with target concurrency 64. The deployment requests 128 running sequences
per replica; the engine reports its actual limit separately (36 in this run).
The hybrid model's memory allocation can lower the requested limit.

The client manifest comes from `replay-clients.json` produced by the preceding
DAPO experiment. It identifies 16 trained models with immutable versions 4 and 7:
32 distinct adapters, rather than 32 independently trained models. Checkpoints
must remain available in the volumes configured in `cache_deployment.py`.

Each adapter has 16 independent closed-loop streams, divided among four shared
prefix groups. Each stream submits four requests, starting its next request as
soon as the previous one completes. There is a two-second launch ramp and no
global turn barrier. This gives 512 concurrent clients and 2,048 requests per
policy, with one generation per request. Every prompt contains 8,192 tokens:
7,168 shared within its group, followed by 1,024 unique tokens. Every response
contains 512 tokens; `stop=[]` maps to SGLang's `ignore_eos=True`.

The generator samples valid token IDs from saved DAPO prompts. The resulting
prompts are synthetic; this experiment evaluates serving behavior, not math
quality. The seed is 12026. A hash verifies identical traces across policies.

An eight-vCPU CPU function drives eight independent router-state objects through
the production `sample_task` path and real Modal Flash. It does not emulate eight
separate `execute_sample` containers or their ingress overhead. Destination
selection uses the checked-out router; the generator is not a production routing
controller.

## Controlled state and instrumentation

Before each policy, all 32 adapters are registered on every replica, the same
seven adapters plus the base model are placed in GPU slots, and KV cache is
flushed. Preparation is outside the timer. All arms use the same eight replicas.

`cache_instrumentation.py` derives a separate image from the pinned rollout image.
It fails if the expected SGLang source no longer matches. It counts GPU loads and
evictions at the adapter memory pool and counts a request's GPU LoRA hit once,
immediately before its first prefill batch. Base-model requests are excluded.
Chunked prefill does not count a request repeatedly. The counter does not report
a hit merely because an adapter is registered in CPU memory.

Counters use CPU bookkeeping without CUDA synchronization. Load-call duration is
host elapsed time, summed across replicas; it is neither GPU transfer duration
nor critical-path time saved. The derived sidecar exposes benchmark snapshots and
cache flushing. Production sidecars do not receive these endpoints.

## Run and reproduce

Run from the repository root with the Spindle environment and Modal credentials.
The experiment requires access to the saved adapter volumes. Plotting additionally
requires NumPy and Matplotlib.

```bash
export MODAL_ENVIRONMENT=lilo-deploy
python scripts/benchmarks/lora_routing/cache_deployment.py \
  --run UNIQUE_RUN --output /tmp/cache-run/pools.json
python scripts/benchmarks/lora_routing/cache_monitor.py \
  --root /tmp/cache-run --minutes 60 > /tmp/cache-run/monitor.log 2>&1 &
python scripts/benchmarks/lora_routing/cache_replay.py \
  --pools /tmp/cache-run/pools.json \
  --clients /path/to/replay-clients.json \
  --output /tmp/cache-run/main --sequential
python scripts/benchmarks/lora_routing/cache_report.py /tmp/cache-run/main
touch /tmp/cache-run/STOP
```

The launcher deploys one eight-GPU pool and assigns all policies to it. Shared
pools always run sequentially. The replay stops its pools on exit unless
`--keep-pools` is supplied. The independent monitor bounds their lifetime and
audits resource cleanup. `--arms kv default lora` changes execution order for a
repeat. Do not run two experiments concurrently on the same pool.

The request timeout is 240 seconds, with a 900-second retry budget. Transport
retries remain in latency and throughput measurements. Native admissions must
fall between completed logical requests and recorded physical attempts. Extra
engine admissions from retries are reported as duplicate work. Incomplete runs
are saved with their error and cause a nonzero exit.

## Outputs and definitions

Each arm produces JSON containing every request, physical attempt, output hash,
version check, periodic replica observation, and initial/final counters. The
report writes `summary.json`, `requests.csv`, and PNG/PDF figures for caches,
latency, throughput, replica load, reconstructed engine occupancy, and (for
DAPO replay) client rollouts and adapter working sets. Native prefix lookup
count, failures, and mean/maximum duration are included when recorded.

- KV hit ratio is total cached prompt tokens divided by total prompt tokens.
- Hot-LoRA hit ratio is first-prefill GPU-resident requests divided by physical
  engine admissions, including duplicate work from retries.
- Completion latency includes routing, transport, queueing, prefill, and decode.
- Output throughput is fixed completed output tokens divided by full workload
  duration, including the launch ramp and drain.
- GPU loads and evictions count adapter slot transitions, excluding the base.
- Load CV is the mean coefficient of variation across replica outstanding-request
  counts while at least half the offered clients remain active.

KV, latency, and completion-throughput curves use non-overlapping 20-second
completion cohorts. The throughput curve counts tokens in completed responses;
it is not an instantaneous GPU decode-rate trace. LoRA curves difference native
counters in approximately ten-second admission intervals. Rates over gaps longer
than 20 seconds are omitted; load curves break across gaps longer than 15 seconds.
The report includes observation coverage and maximum gaps. Missing snapshots
limit load-balance comparisons but do not change complete request records or
initial/final counter totals. Overall averages include initial cold cache and
the tail.

Validation checks exact prompt/output lengths, finite aligned log probabilities,
length-based completion, exact served adapter versions, route identity, identical
trace hashes, the same eight replicas, no pending requests at completion, zero
CPU registrations during measurement, and bounded native admission accounting.
These checks do not establish bitwise numerical equivalence across batch shapes.

The fixed pool deliberately isolates routing. This is not an autoscaling test,
training-throughput measurement, or estimator of statistical significance from
one trial. The LoRA router still uses approximate recent/active-adapter hints;
benchmark GPU counters are observations and are not fed into routing decisions.


## Independent DAPO client replay

Build a manifest from a completed 16-client training run:

```bash
python scripts/benchmarks/lora_routing/dapo_manifest.py /path/to/training-run \
  --bulletin-volume routing27-bulletin-1011 --output /tmp/cache-run/dapo-clients.json
python scripts/benchmarks/lora_routing/cache_replay.py \
  --pools /tmp/cache-run/pools.json --clients /tmp/cache-run/dapo-clients.json \
  --output /tmp/cache-run/dapo --workload dapo \
  --groups 16 --samples 8 --rounds 3 --stagger-seconds 5 --decode-tokens 512 \
  --arms lora kv --sequential --keep-pools
python scripts/benchmarks/lora_routing/cache_report.py /tmp/cache-run/dapo --arms lora kv
```

This uses the real tokenized math problems and two saved trained checkpoints per
client. It runs eight separate nonpreemptible CPU functions, each with two
independent clients and one pool router, through production `sample_task`. The
coordinator waits once for all eight workers to be ready, then releases a common
start timestamp. There is no coordination barrier between training rounds.

Each client submits 16 groups × 8 samples (128 concurrent requests per rollout),
then starts its next rollout while its recorded optimizer/publication delay
elapses. The third rollout uses the second checkpoint. Problems advance between
rounds; every sample of the same problem has the same affinity key, scoped to its
sampling session and exact version. Temperature is 1.0, seeds are fixed, and
`ignore_eos=True` fixes each decode at 512 tokens. The default trace has 6,144
requests and 3,145,728 output tokens, with at most 2,048 concurrent requests.
Actual concurrency is measured, since staggered starts, independent progress,
and update intervals reduce overlap.

**This replays inference on GPUs; update intervals are recorded delays, not new
optimizer work. It measures request and rollout completion, not live training
throughput or model quality.** The source training configuration is retained in
the original run. Resetting GPU adapters and KV state also clears prefix hints.

`--stagger-seconds` controls client arrival density without introducing round
barriers. `--hot-samples 128` changes client 0 to one 128-sample problem per
rollout, keeping its total request count unchanged; other clients remain 16 × 8.
This is an explicit high-fanout stress case, not the source job's 8-sample group.

The report adds per-client rollout timelines and active adapter working sets.
Busy-interval balance metrics use total actual engine running capacity as their
threshold (288 here), and report the number of eligible snapshots. All transport
retries remain in the measured result. Nonpreemptible CPU workers match
production sampling workers; GPU allocation is unchanged.


For a fixed-arrival pressure window, use the same DAPO command with
`--rounds 1 --stagger-seconds 0.5`. This submits one ordinary 128-sequence rollout
from each of 16 clients over 7.5 seconds: 2,048 requests and 1,048,576 output
tokens. There are no later submissions whose timing could be changed by a
transport retry. All clients remain independent; it is a concurrent rollout
window, not a measurement of a complete optimizer cycle. The second checkpoint
is registered for equal initialization but is not requested in this variant.

`reconstructed-engine-load` estimates successful-attempt occupancy by anchoring
engine-reported e2e duration at its response-sent timestamp, sampled every second.
SGLang's e2e finish occurs just before response sending, so the inferred interval
is shifted by that gap. Failed attempts without responses cannot be reconstructed;
report their count alongside this figure. This supplements incomplete native
snapshots and is not an exact native scheduler queue trace.


The mixed math case uses `--groups 16 --samples 8 --large-group-clients 8
--rounds 1 --stagger-seconds 0.5`. Clients 0–7 reshape their unchanged 128-sequence
budget into one problem with 128 samples; clients 8–15 retain 16 problems with
eight samples each. The trace still has 2,048 requests and 1,048,576 output tokens.
This represents mixed group-size experiments sharing an inference pool. It is
an explicit workload variant, not the source DAPO job's default configuration.
[Published MATH GRPO experiments](https://arxiv.org/html/2603.01162v3) evaluate
128-sample groups; this benchmark does not reproduce that paper's optimizer work.

The final router resolves native Flash affinity once per unseen prefix, before
request fanout. Raw results include lookup count/failure/time aggregates and the
resolved homes. Measured request latency includes this lookup. The benchmark-only
sidecar now gives engine observations a separate HTTP connection pool so they do
not queue behind its generation connections; production request handling is
unchanged by this instrumentation change.

## Timed measured-work comparison

The measured-work experiment enables `SGLANG_SPINDLE_ROUTING_OBSERVATIONS=1` in
all arms so observer overhead is shared. Unlike the historical router, `work`
uses actual supported engine KV boundaries and CPU/GPU LoRA inventory.
`work-no-adapter` uses the same grouping and cache/load score but omits both
adapter setup terms. The routing profile must come from calibration on the same
model, rank, hardware, context and concurrency regime. The calibration script
requires the benchmark image's idle-only reset endpoints.

After deploying the pool and starting the independent monitor as above:

```bash
python scripts/benchmarks/lora_routing/work_calibrate.py \
  --pools /tmp/cache-run/pools.json \
  --clients /path/to/dapo-clients.json --output /tmp/cache-run/calibration

# Warmed reuse: 16 clients, each 8 groups × 8 samples, all using version 4.
python scripts/benchmarks/lora_routing/cache_replay.py \
  --pools /tmp/cache-run/pools.json --clients /path/to/dapo-clients.json \
  --output /tmp/cache-run/warmed --workload dapo \
  --groups 8 --samples 8 --rounds 100 --duration-seconds 300 \
  --stagger-seconds 0.5 --decode-tokens 512 \
  --long-prefix --repeat-prefixes --warm-prefixes --fixed-version \
  --work-rates /tmp/cache-run/calibration/work-rates.json \
  --arms kv work --sequential --keep-pools
python scripts/benchmarks/lora_routing/cache_report.py \
  /tmp/cache-run/warmed --arms kv work

# Version turnover with mixed group sizes and one hotter client.
python scripts/benchmarks/lora_routing/cache_replay.py \
  --pools /tmp/cache-run/pools.json --clients /path/to/dapo-clients.json \
  --output /tmp/cache-run/turnover --workload dapo \
  --groups 8 --samples 8 --large-group-clients 8 --hot-samples 128 \
  --rounds 100 --duration-seconds 300 --stagger-seconds 0.5 --decode-tokens 512 \
  --long-prefix --repeat-prefixes --warm-prefixes --cold-versions \
  --work-rates /tmp/cache-run/calibration/work-rates.json \
  --arms kv work work-no-adapter --sequential --keep-pools
python scripts/benchmarks/lora_routing/cache_report.py \
  /tmp/cache-run/turnover --arms kv work work-no-adapter
touch /tmp/cache-run/STOP
```

`--duration-seconds` bounds new rollout admissions, not final drain. Each client
advances independently; the 100-round cap is not a fixed request count. The
report includes both full-run throughput and completed tokens in [60,300).
Full-run duration includes final recorded update delays and all retries. Faster
policies may issue more rollouts before admission closes.

`--long-prefix` substitutes 8,192 synthetic input tokens with a 7,168-token prefix
reused within client/problem/version and a 1,024-token changing suffix.
`--warm-prefixes` runs three serialized suffixes per problem through native KV
affinity before the timer. In the turnover command, client 0 submits 1 × 128;
clients 1–7 submit 1 × 64; clients 8–15 submit 8 × 8. Maximum simultaneous
requests are 1,088. The first two rollouts use version 4 and later rollouts use
version 7. `--cold-versions` unregisters all identities before priming only
version 4, so version 7 requires CPU registration. Checkpoint files already
exist; this does not guarantee network-cold Volume reads. Registration counts
are expected and validated in this case, rather than required to be zero.

The grouped router splits a known sample group at measured decode capacity
(36 here), not the Flash target. The profile records prefill/decode rates,
GPU-load and exposed-registration delays, and this capacity. Calibration uses
serialized reuse controls, equal-cache paired adapter measurements, and cohort
engine makespan. Its output hashes validate the serialized control only, not
universal bitwise equivalence across batching schedules.

The October 13 report preserves the failed individual-placement implementation
at `6be8811` and grouped revision at `283d689`. The grouped warmed comparison
reuses the earlier KV baseline; its raw trace hash differs because the config
added `decode_concurrency`. Workload fields excluding `work_rates` and replica
identity sets were explicitly checked equal before plotting that comparison.
The turnover arms have matching trace hashes. Do not treat reused baselines or
closed-loop request counts as independent randomized replicates.
