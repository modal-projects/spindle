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

Unlike FPRS+'s request-history index and weighted cost, this policy uses engine
observations and explicit residency/load gates. It follows the SGLang cache-aware
policy's separation of load balancing and prefix preference, adding exact LoRA
version identity and residency tiers. It does not implement either system wholesale.
Inspected sources: Modal FPRS+ `89381450167ff47250d9ac57ec99f1bdab8ccbfc`,
`crates/flash-proxy/src/bin/fprs-plus/router/cache_aware.rs`; SGLang
`a270ed4ed5bf9379ebae8859da3fe3e374e637f4`,
`sgl-model-gateway/src/policies/cache_aware.rs`.

## Execution and limits

- `rust/router/src/policy.rs`: policy selection only.
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
