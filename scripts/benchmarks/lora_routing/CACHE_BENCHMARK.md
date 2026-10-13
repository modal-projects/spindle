# Fixed-work LoRA and KV cache comparison

The measured results and plots are in [the Notion report](https://app.notion.com/p/3f71e7f1694981628893fea0f75f2711?pvs=204)
and [the repository results](../../../docs/lora-cache-routing-results.md).

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
report writes `summary.json`, `requests.csv`, and six PNG/PDF figures.

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
