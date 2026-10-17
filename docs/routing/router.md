# Inference routing

`sample_task` prepares one request per sample group. `sample_one` calls
`await router.select_destination(request, attempt_id, deadline=...)`, sends to
that destination through Flash, and calls `router.complete(...)` in `finally`.
Sampling does not inspect catalogs, scores, or reservations.

## Policies

The default `inference_lora_routing = "affinity"` retains native Flash routing.
The Rust policies share observation freshness, admission and outcome handling:

| Policy | Selection |
| --- | --- |
| `joint` | Earlier weighted remaining-work estimate in seconds; retains sibling destination while capacity permits. |
| `least_load` | Earlier request-count policy, also retaining sibling destination. |
| `load_balanced` | Lowest active count, including unobserved local reservations; no sibling override. Admission-only control. |
| `lora_radix` | Load band, adapter residency tier, longest confirmed prefix, sibling tie, active count, deterministic tie. |
| `cache_wait` | Confirmed prefix preference within a load band; bounded waits for a full cached replica or a sibling establishing its prefix. |
| `cache_aware` | Pool-wide imbalance switches to least-load; otherwise prefer a substantial prefix match, with adapter residency breaking ties. |

For `lora_radix`, let `m` be the minimum active count among eligible replicas.
Exclude a replica with count `n` from affinity preference when **both**
`n - m > inference_routing_balance_absolute` and
`n > m * inference_routing_balance_relative`. Defaults are 4 and 1.25;
these are experiment parameters, not calibrated optimal constants.
Within this band, prefer an adapter already on GPU, then one pending/loading,
then one registered on CPU, then an unregistered adapter. Within that tier,
maximize confirmed reusable prefix tokens. Sibling affinity only breaks ties;
it cannot override the load band or a longer confirmed prefix.

This allows a hot adapter to spread when its current replicas become overloaded.
No replica is a permanent home. Engine eviction controls residency; the router
does not pin adapters or explicitly shrink their placement.
A strict residency gate can choose a warm adapter with little KV over a cold
adapter with substantial KV. That is a deliberate limitation to evaluate,
not an assertion that loading an adapter always costs more than prefill.

`cache_aware` uses the public SGLang policy's pool-wide switch. If both
`max_active - min_active > balance_absolute` and
`max_active > min_active * balance_relative`, select the least-loaded eligible
replica, breaking ties by prefix length, then adapter residency. Full replicas
remain part of this imbalance calculation even though they cannot receive work.
Otherwise, if the best eligible prefix covers more than
`inference_routing_cache_threshold` of the prompt (default 0.5), select the longest
match, then adapter residency, then load. Below that threshold, select least-load
with residency breaking ties. This policy has no sibling preference.

For controlled comparison, the load thresholds remain 4 and 1.25, as in
`lora_radix`; these are not claimed to be SGLang defaults or tuned optima.
Unlike SGLang's request-history index, this policy uses observed token prefixes
scoped to an exact adapter version. It keeps the same local admission and polling
as the other Rust policies. It does not wait for a full replica solely to retain
its prefix, and does not predict when an adapter load is cheaper than prefill.

## Bounded cache waiting

`cache_wait` considers fresh full replicas before admission. It prefers the
longest confirmed prefix above the 50% threshold within the same 4/1.25 load
band. Adapter residency breaks ties. Unlike `cache_aware`, an unrelated overloaded
replica does not turn off cache preference throughout the pool.

If the better cached replica is full, wait for at most
`min(2 seconds, extra_cached_tokens / prefill_tokens_per_second)`. The alternative
is the best confirmed cache on any available replica. Spill on expiry, excess
load, stale observations or failure. No request is dispatched above capacity.
The measured prefill rate is an approximation; this is a bounded experiment,
not an accurate latency predictor for all batches.

The first sibling also creates a group-local pending-prefix hint. Other siblings
may wait up to two seconds from its dispatch for the engine to report the full
aligned prompt prefix. This grace covers observation delay; it can add latency
when the engine does not retain that prefix. It ends on leader completion,
failure, overload, or expiry. Pending work never counts as a confirmed hit.
Independent groups and adapter versions do not share this hint.

Affinity elapsed time is local to the asynchronous selection call and cannot
reset on polling or target changes. Waiting for a completely full pool does not
start it. The wrapper wakes at the remaining budget even when it is shorter
than the normal poll interval. After release, routing uses confirmed cache and
load; no permanent LoRA home is created.

Benchmark completion records retain group ID, predicted cached tokens, best
observed cache, eligible replica count and routing wait separately from the
engine's actual cached tokens. The same routing fields are exported on OTLP
attempt spans. Missing observations are not cache misses.

## Prefix index

`rust/router/src/radix.rs` holds one compressed radix tree per exact adapter
version. Edges contain cumulative 128-bit fingerprints at 64-token boundaries;
each boundary records the replicas with reusable device state there. One traversal
returns the longest confirmed match for each replica. This is probabilistic hash
identity, not a collision-free token comparison. Hashes are routing hints only;
SGLang still performs the actual cache lookup and exact-version validation.

The engine observer exports parent-linked boundaries, capped at 8,192 nodes per
replica while preserving complete ancestry. Invalid hybrid/Mamba boundaries can
be structural nodes but receive no reuse credit. Evicted ancestors, host-only
cache and unsupported cache types receive no credit. Snapshot replacement removes
old observations. New requests never manufacture confirmed cache entries.
The last prompt token is excluded because it must be evaluated for logits.

The public SGLang cache-aware policy motivates separating load balancing from
prefix preference. This policy applies the absolute/relative load test to each
candidate rather than switching the whole pool to least-load routing. It adds
exact LoRA version identity, adapter residency tiers, and observed reusable
engine state. It does not implement SGLang's router wholesale.
Public reference: [SGLang cache-aware policy](https://github.com/sgl-project/sglang/blob/a270ed4ed5bf9379ebae8859da3fe3e374e637f4/sgl-model-gateway/src/policies/cache_aware.rs).

## Execution and limits

- `rust/router/src/policy.rs`: immediate policy selection.
- `rust/router/src/cache_wait.rs`: bounded affinity decisions.
- `rust/router/src/lib.rs`: request hashing, immutable snapshots, reservations,
  capacity checks and outcome feedback.
- `src/spindle/inference/router.py`: background refresh and asynchronous admission.
- `src/spindle/providers/modal/router.py`: discovery and Flash addresses.

One router lives in each sampling-worker process per inference pool, shared
across adapters. There is no central request coordinator. Snapshot parsing and
radix construction run in a background thread with the GIL released; installation
swaps immutable state. Selection is a short synchronous Rust call under the GIL.
This is not lock-free concurrent selection within one process.

Discovery runs every 0.5 seconds, membership refreshes every 5 seconds, and engine
observations are produced at most once per second. Observations older than three
seconds are ineligible. Local reservations cover requests not yet reported by the
engine. Other workers can race because reservations are not globally coordinated;
SGLang owns final admission. 503/429 means busy, not unhealthy. Saturation waits
locally with a deadline and cancellation, without native routing fallback.

All Rust policies currently require `sglang_cfg.max_running_requests` and
`inference_routing_work_rates` (prefill/decode tokens per second, GPU-load seconds,
registration seconds). The gated policy does not use these rates to rank replicas;
they remain diagnostic estimates and support the older weighted comparator.
Latest-version sampling retains native routing because its exact version resolves
downstream. Actual cache reuse can fall below observed reuse because of eviction
between observation and arrival; snapshot omissions can also understate reuse.

Build with `maturin build --release --manifest-path rust/router/Cargo.toml`.
Installing Spindle from source builds the extension through setuptools-rust.
Modal builds it in the frontend image; engines do not need this router extension.
