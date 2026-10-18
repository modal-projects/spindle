# Native KV affinity with bounded LoRA-aware overflow

The current router preserves Flash's native prefix home and spills requests when
that home's excess backlog exceeds a configured budget. On one paired replay of
mixed DAPO sampling groups, this reduced mean request latency **6.86%**, p95
**20.72%**, and GPU LoRA loads/evictions **19.05%** versus native KV affinity.
It substantially reduced uneven replica load. **It did not improve full-workload
throughput:** read-timeout retries dominated the final drain, and throughput was
0.41% lower. KV hit ratio also decreased.

This is evidence for a useful overload policy, not a general replacement for KV
affinity. The final version still needs a full long-prefix rerun, repeated order
reversals, and live optimizer-throughput validation. The preceding long-prefix
revision regressed mean latency and adapter churn; those results are below.

Measured routing/harness commit: `965149f`. Documentation-only successor:
`a5ca2e6`. [Draft PR #52](https://github.com/modal-projects/spindle/pull/52).
The [Notion report](https://app.notion.com/p/3f71e7f1694981628893fea0f75f2711)
contains the six primary plots and historical results. The original three-policy
benchmark is [separate](lora-cache-routing-results.md).

## Implemented design

There is one router per inference pool per sampling process, shared across its
LoRAs. The request path remains `sample_task` → authenticated Modal Flash → exact
replica sidecar → SGLang. No application-wide scheduling controller is introduced.

1. Refresh pool observations. Membership is cached for five seconds; replica
   observations refresh at most once per second per active router. Concurrent
   refreshes share one task. Fresh requests use the previous snapshot immediately;
   initial/stale discovery can wait up to three seconds before Flash fallback.
2. Prepare the exact adapter/version and prefix key. A previously unseen prefix
   performs one authenticated `GET /routing-state` with the same `Modal-Session-ID`
   as native KV generation. This learns Flash's actual home. Concurrent callers
   for that prefix share the lookup; other prefixes proceed independently. The
   lookup is bounded to two seconds. Failure falls back to native Flash.
3. Each physical generation calls synchronous `select_destination`. For host h,
   estimate outstanding work as `max(reported[h] + new_local_arrivals[h], local[h])`.
   New local arrivals exclude reservations already present at observation time.
   Eligible hosts have load below `min(loads) + target_concurrency` (64 here).
4. Keep the prefix's home while it is eligible, even if its adapter is cold.
   If overloaded, prefer eligible replicas with the same exact-version prefix
   hint, then an active/recent exact adapter, then room in their active adapter
   working set. Choose least load within that preference, with deterministic
   ties. If all working sets are full, use the less-loaded of two prefix-hashed
   candidates. Temporary overflow does not rewrite the prefix's home.
5. Reserve locally before dispatch; release on completion, errors, and
   cancellation. Explicit dispatch uses `Modal-Flash-Upstream` through Flash.
   Exact adapter-version validation remains the sidecar's responsibility.

The budget is an interpretable request-count allowance for locality, **not** a
calibrated time model or a hard distributed admission bound. Different workers
can overshoot between polls. Prefix hints represent recent use, not guaranteed
KV residency; active/recent adapter hints do not guarantee GPU residency. Neither
benchmark GPU counters nor synthetic future-request knowledge enter the policy.
The 8,192-entry frontend home map and 4,096-entry sidecar prefix map bound state;
sidecar hints expire after 120 seconds. See [the design](lora-routing.md) and
[router source](../src/spindle/inference/lora_routing.py).

The measured run made **136 native prefix lookups**, with zero failures, mean
**181.86 ms** and maximum **432.54 ms**. Those costs are included in latency.
Warm-prefix destination selection makes no network call and takes no lock; its
CPU execution cost was not separately microbenchmarked. Eight active routers
polling eight replicas can issue about 64 observation requests/s in this setup.
All 133 prefix homes unaffected by LoRA retries matched the baseline's observed
native destination; three retry-affected keys were excluded from that check.

## Primary workload and controls

This is a real-GPU **inference replay**, using real tokenized DAPO math problems
and immutable trained checkpoints from the earlier independent 16-client run.
It does not run new optimizer work or establish math quality/training throughput.

| Setting | Value |
| --- | --- |
| Model / adapters | Qwen/Qwen3.8-27B; rank 32; 16 independently trained models |
| Hardware | Same eight H200 replicas for both arms; TP=1; us-west |
| Replica resources | 8 vCPU, 64 GiB host RAM; static GPU-memory fraction 0.8 |
| Pool | Fixed min=max=8; target concurrency 64; autoscaling disabled |
| Engine | SGLang v0.5.17, commit `d050d06437d96196fc68d5b4e5c246408790d537` |
| Engine scheduling | lpm; 8,192-token chunked prefill; context 16,384; max queued 4,096 |
| Actual running capacity | 36 per replica / 288 total; 128 requested but constrained by hybrid-model memory |
| Kernels / dtype | fa3 attention, csgmv LoRA; model and KV dtype auto; no quantization; radix cache enabled; page size 1 |
| LoRA target modules | q/k/v/o_proj, gate/up/down_proj |
| LoRA capacity | 8 GPU slots; 256 CPU registrations; overlap loading off |
| Registered / requested versions | 32 versions CPU-registered per replica; this one-rollout test requests 16 version-4 identities |
| Clients / frontends | 16 clients in eight actual nonpreemptible CPU containers; two clients and one router per container; 2 vCPU / 8 GiB each |
| Group distribution | Clients 0–7: one problem × 128 samples. Clients 8–15: 16 problems × 8 samples. Every client submits 128 generations. |
| Arrival schedule | One startup readiness handshake; clients start 0.5 seconds apart over 7.5 seconds; no inter-client rollout barrier |
| Work / concurrency | One rollout per client; 2,048 requests; maximum 2,048 concurrent requests; observed peak 2,048 |
| Prompt / output lengths | Real prompts 124–243 tokens; 408,960 input tokens total; exactly 512 output tokens/request, 1,048,576 total |
| Sampling | Temperature 1; deterministic seed schedule starting at 4242; `stop=[]` → `ignore_eos=True` |
| Affinity | Stable problem key scoped by the client's sampling session and exact adapter version |
| Transport | Identical production TCP keepalive; 512 frontend connections / 256 keepalive; 240-second request timeout; 900-second retry budget |
| Order / repetitions | Native KV then final router; one paired trial on the same physical replicas |
| Reset | Drain; register all 32 versions; identical seven-adapter-plus-base GPU warmset; flush KV and prefix hints before each arm |

The source training configuration uses 16 groups × 8 samples. The 128-sample
half of this workload is an explicit larger-group variant, not a claim about the
source job's default. It models a shared serving pool with heterogeneous math
sampling jobs: a few large same-problem fanouts collide with ordinary groups.
Groups through 128 have been evaluated for MATH GRPO in
[Demystifying GRPO, section 5.2/table 2](https://arxiv.org/html/2603.01162v3);
this replay does not reproduce that paper's optimizer or batch configuration.

Unlike a hand-selected overloaded host, every prefix starts from its actual
native Flash destination. The same client IDs, prompts, seeds, checkpoints,
arrival schedule, and cache reset are used in both arms; trace hashes match.

## Results: native KV versus final router

| Metric | Native KV | Final router | Change |
| --- | ---: | ---: | ---: |
| Mean request completion | 61.165 s | 56.968 s | −6.86% |
| p50 completion | 56.733 s | 56.277 s | −0.81% |
| p95 completion | 123.642 s | 98.028 s | −20.72% |
| p99 completion | 137.436 s | 108.511 s | −21.05% |
| Mean client rollout completion | 117.486 s | 107.785 s | −8.26% |
| p95 client rollout completion | 252.130 s | 252.483 s | +0.14% |
| Full workload duration | 256.722 s | 257.770 s | +0.41% |
| Full workload output tokens/s | 4,084.48 | 4,067.88 | −0.41% |
| Mean successful engine attempt | 34.081 s | 34.045 s | essentially unchanged |
| Mean total minus successful engine duration | 27.084 s | 22.924 s | −4.161 s |
| KV token hit ratio | 50.56% | 43.52% | −7.04 percentage points |
| First-prefill hot-LoRA hit ratio | 87.95% | 86.99% | −0.96 percentage points |
| Hot-LoRA hits / misses | 1,803 / 247 | 1,785 / 267 | includes duplicate admissions |
| GPU loads / evictions | 63 / 63 | 51 / 51 | −19.05% each |
| Summed host load-call time | 4.243 s | 3.562 s | −0.680 s; not GPU critical-path savings |
| Average replicas per adapter | 4.125 | 3.438 | −16.67% |
| Average replicas per prefix group | 1.015 | 1.206 | deliberate overflow increases replication |
| Mean busy-snapshot replica load CV | 0.606 | 0.211 | −65.19% |
| Busy snapshots with an idle replica | 23/49 (46.94%) | 2/46 (4.35%) | −42.59 percentage points |
| Busy replica observations with >8 outstanding adapter versions | 19.13% | 0% | lower working-set pressure |
| Extra physical attempts / engine admissions | 2 / 2 | 4 / 4 | more retry work in final router |

Both arms completed all 2,048 logical requests. Native engine counter totals are
2,050 and 2,052 admissions respectively, consistent with those retries. There
were zero measurement-time CPU registrations, route mismatches, or engine
retractions. Explicit destination selection covered 99.85% of router attempts;
remaining attempts used Flash fallback. All retries remain in every duration.

The balancing failure is visible in placement and drain. Baseline replica totals
range from **128 to 368** successful requests, versus **222 to 281** with overflow.
Native KV's lighter replicas drain while heavily assigned replicas still have
work. The router keeps replica backlogs closer together and drains the main
burst earlier, but timeout/retry stragglers keep full makespan near 257 seconds.
It reduces, rather than eliminates, idle replicas during busy intervals.

| Client class | Native mean / p95 request latency | Router mean / p95 |
| --- | ---: | ---: |
| 128-sample problems (1,024 requests) | 45.944 / 97.264 s | 42.480 / 83.469 s |
| Ordinary 8-sample problems (1,024 requests) | 76.386 / 133.858 s | 71.457 / 104.567 s |

This helps both the large-group jobs and ordinary clients sharing their pool.
The mechanism is primarily better placement/backlog balance, not faster kernels:
successful engine duration is unchanged. The difference outside that duration
also includes transport, sidecar waiting, routing, and failed-attempt time; it
cannot be assigned entirely to Flash queueing. Lower load-call host time alone
does not explain a four-second mean latency improvement.

Fewer GPU transitions can coexist with a slightly lower first-prefill hit rate.
Several requests can first encounter the same cold adapter before one load
serves their batch, and an adapter can move again after first prefill. These
metrics have different denominators and measure different behavior. The price
of overflow is additional prefix replication and lower KV reuse.

## Iterations and the long-prefix control

All numbers below include retries. Each row compares that revision to native KV;
they are not independent repetitions of the final design.

| Revision / workload | Native mean → router mean | Native wall → router wall | GPU loads, native → router |
| --- | ---: | ---: | ---: |
| Pool-mean eligibility, three DAPO rollouts (`ee2aee8`) | 44.787 → 61.284 s | 428.529 → 450.892 s | 317 → 285 |
| Additive budget, model-first placement, same three-rollout trace (`7026d20`) | 44.787 → 44.959 s | 428.529 → 563.744 s | 317 → 141 |
| Additive/model-first, ordinary burst (`7026d20`) | 55.504 → 65.023 s | 127.097 → 199.629 s | 110 → 35 |
| Custom KV hash first, ordinary burst (`823db7d`) | 55.504 → 58.529 s | 127.097 → 168.932 s | 110 → 100 |
| Two-choice custom hash, ordinary burst (`ec54404`) | 55.504 → 57.134 s | 127.097 → 258.197 s | 110 → 103 |
| Two-choice custom hash, long-prefix control (`ec54404`) | 42.997 → 45.713 s | 343.789 → 327.258 s | 278 → 528 |
| Native home + LoRA-aware overflow, mixed groups (`965149f`) | 61.165 → 56.968 s | 256.722 → 257.770 s | 63 → 51 |

The second row reuses the first row's baseline; the fourth and fifth reuse the
third row's baseline. No independent-repeat claim is made. The ordinary burst
uses 16 clients × 16 groups × 8 samples, one rollout, 0.5-second client staggering,
and 512 decode tokens. Three-rollout runs use the same per-rollout batch, 5-second
staggering, checkpoint sequence 4/4/7, and recorded independent update/publication
delays of roughly 34–96 seconds. Each client prefetches its second rollout while
its update delay elapses; there is no cross-client step barrier. Those runs have
6,144 requests and 3,145,728 output tokens each.

The long-prefix control uses all 32 adapter versions, 16 streams per version,
four conversations per version, four turns per stream, 512 concurrent closed-loop
streams and 2,048 total requests. Each prompt is 8,192 tokens (7,168 shared and
1,024 unique), with 512 decode tokens, seed 12026, and a two-second launch ramp.
It uses one 8-vCPU driver with eight independent router objects, not eight real
frontend containers. It ran sequentially on the same second eight-GPU pool.

That preceding revision lowered p95 from 140.55 to 108.30 seconds but **increased
mean latency 6.32% and GPU loads 89.93%**. KV hit ratio fell from 74.10% to 67.13%;
hot-LoRA hit ratio fell from 80.31% to 63.40%. There were seven retries for KV
versus one for that router, materially affecting wall time. It is a regression
control, not evidence that the final native-home policy handles long contexts.

Other completed pilots included a 128-request fixed-length correctness replay,
a 192-request real-frontend independent-progress replay, and an all-large-group
2,048-request diagnostic using the custom-hash revision. They did not establish
a final-policy win. Incomplete runs caused by initial transport behavior, CPU
preemption/startup timing, and a cleanup-boundary interruption were preserved
but excluded from comparative results. Nonpreemptible workers, a readiness
handshake, and matched production transport settings resolved the setup defects.

The observed failures explain the final design: a pool-mean cutoff fragmented
groups; model-first placement manufactured hot queues even when adapter capacity
was plentiful; custom hash placement changed the baseline distribution. Keeping
the native home avoids those initial-placement changes. Overflow is the only
normal reason to trade an established home's KV reuse for another host.

## Measurement, correctness, and remaining work

KV hit ratio is token-weighted. Hot-LoRA hits are counted once per physical engine
request immediately before first prefill, using GPU-residency instrumentation.
GPU loads/evictions count actual non-base adapter-slot transitions. The summed
load-call timer is unsynchronized host elapsed time, not device transfer time.
Native outstanding counts include sidecar requests waiting to reach the engine.
Busy snapshots require at least 288 outstanding requests, the pool's actual
running capacity. Both final arms have 125 complete snapshots, zero failed
snapshots, and maximum gaps of 2.31/2.61 seconds; coverage within ten seconds is
100%. Successful-attempt engine occupancy is also reconstructed, explicitly
approximate and excluding failed attempts without responses.

Time plots use 20-second completion cohorts for KV, latency, and completed-token
throughput; adapter hit/eviction rates use native counter differences over about
ten seconds. Completed-token throughput is not instantaneous GPU decode rate.
The full trace, including cold starts and drain, determines aggregate metrics.

**67 focused tests passed** across router selection/reservations, native-affinity
lookup coalescing and cancellation, HTTP headers, sidecar version/prefix handling,
sampling, and benchmark instrumentation. Live checks verify exact served versions,
512-token outputs, finite aligned log probabilities, length-based termination,
route identity, matched traces and starting caches, admission accounting, and
drained replicas. These checks do not establish bitwise equivalence across
different batch shapes; a concurrent greedy pilot did not match every output.

The final long-prefix rerun was not performed before the bounded cleanup deadline.
The attempted monitor extension was blocked by automatic approval review because
two cleanup monitors could disrupt the pool; no user approval arrived for an
extension. Both pools stopped successfully. The maximum concurrent allocation
was **16 H200s**, with zero GPUs active afterward.

Before enabling broadly: repeat the final mixed trial in reverse order, run the
final policy on the full long-prefix and ordinary eight-sample controls, test
heterogeneous decode costs and changing adapter popularity, investigate the
remaining transport stalls, and measure live independent training step time.
Autoscaling is unverified: good balance on a fixed pool does not establish correct
Flash scale-out behavior. Keep this PR in draft until those checks succeed.

## Reproduce

Use the deployment, monitor, and immutable-checkpoint manifest instructions in
[CACHE_BENCHMARK.md](../scripts/benchmarks/lora_routing/CACHE_BENCHMARK.md). With
an eight-GPU pool prepared and monitored, the primary comparison is:

```bash
python scripts/benchmarks/lora_routing/cache_replay.py \
  --pools /tmp/cache-run/pools.json --clients /tmp/cache-run/dapo-clients.json \
  --output /tmp/cache-run/native-mixed --workload dapo \
  --groups 16 --samples 8 --rounds 1 --large-group-clients 8 \
  --stagger-seconds 0.5 --decode-tokens 512 --arms kv lora --sequential
python scripts/benchmarks/lora_routing/cache_report.py \
  /tmp/cache-run/native-mixed --arms lora kv
```

Local evidence: `analysis/kv-lora-training-20261013/` in the parent workspace,
including per-arm JSON/CSV, plots, policy/native-home audit, run metadata and
combined resource audit. Raw request traces and checkpoint identifiers are not
exported to the report.
