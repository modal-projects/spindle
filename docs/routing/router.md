# LoRA-aware routing

Native Flash affinity remains the default (`inference_lora_routing = "affinity"`).
Set `inference_lora_routing = "cache_wait"` to enable the Rust LoRA-aware router
for exact-version LoRA sampling. Latest-version and base sampling retain native
routing because their exact adapter version is not known at selection time.

Configure `sglang_cfg.max_running_requests` and `inference_routing_work_rates`
with positive `prefill_tokens_per_second` and `decode_tokens_per_second`, and
nonnegative `gpu_lora_seconds` and `registration_seconds`. Calibrate these for
the deployment: only the prefill rate controls waiting; the remaining rates
provide diagnostic work estimates, not selection scores.

## Selection

Among replicas with fresh observations, count active requests plus unobserved
local reservations. Only replicas below capacity can receive a request. Let
`m` be the minimum count among available replicas. Exclude a candidate from
cache preference when both `active - m > inference_routing_balance_absolute`
and `active > m * inference_routing_balance_relative` (defaults: 4 and 1.25).

Within that load band, prefer the longest observed prefix if it covers more than
`inference_routing_cache_threshold` of the prompt (default: 0.5). Otherwise choose
least load. Adapter residency breaks ties: GPU, pending registration/load, CPU,
then absent. Group preference and a deterministic hash resolve remaining ties.

If a full, balanced replica has more cached tokens than any available replica,
wait at most `min(2 seconds, extra_cached_tokens / prefill_tokens_per_second)`.
On expiry, route to an available replica. A completely full pool waits for
capacity without starting the affinity budget. Polling and target changes do not
reset an already-started budget. These estimates bound delay; they do not predict
end-to-end latency or guarantee a performance improvement for every workload.

Samples sharing one request group may also wait up to two seconds for a dispatched
sibling's prefix to appear in an observation. The hint expires on leader completion,
failure, overload, or elapsed budget. It never counts as a confirmed cache hit.
Separate single-sample calls do not share this pending-prefix hint.

## Prefix matching

The engine observer exports reusable device-cache boundaries scoped to the exact
LoRA identity and weight version. The Rust index stores compressed radix trees
over cumulative 128-bit fingerprints at 64-token boundaries. One traversal returns
the longest observed match on each replica. The final prompt token is excluded
because it must be evaluated for logits. Hashes are hints; SGLang performs the
actual cache lookup and version validation.

Catalogs retain at most 8,192 boundaries per replica with complete ancestry.
Hybrid/Mamba boundaries without reusable state are structural nodes only.
Evicted ancestors, host-only state, unsupported caches, and different adapter
versions receive no credit. Replacing snapshots removes evicted observations;
dispatching requests never manufactures cache entries.

The rollout image installs read-only cache export on the pinned SGLang fork.
Snapshots are produced at most once per second by the serving scheduler. A bounded
background writer performs file I/O. The sidecar exposes `/routing-state`; no
benchmark endpoint or per-completion archive is needed for routing.

## Execution and limits

Each sampling-worker process maintains one router per inference pool, shared across
adapters. There is no central request coordinator. Python handles discovery and
asynchronous waiting; Rust handles prefix matching, selection, and reservations.
Snapshot parsing and index construction release the GIL off the asyncio thread.
Selection is synchronous under the GIL.

Membership refreshes every five seconds and snapshots every half second.
Observations older than three seconds are ineligible. Reservations only cover the
local frontend; other frontends can race against the same worker snapshot.
SGLang remains responsible for final admission. HTTP 429/503 causes a short busy
cooldown; transport failures cause a longer health cooldown. Saturation and missing
observations wait until the request deadline, without native routing fallback.
Completion, failure, and cancellation release local reservations.

Routing decisions and wait time are recorded on OTLP sampling attempt spans.
Predicted cached tokens can differ from actual reuse due to eviction or snapshot
omissions. The load and cache thresholds are configurable heuristics.

## Build and test

Installing Spindle builds `spindle_router` through setuptools-rust. The frontend
image builds the same extension with maturin. Source distributions and wheels
include the Rust sources needed for subsequent Modal image builds.

Run `cargo test --manifest-path rust/router/Cargo.toml` and the Python inference,
provider, and telemetry tests. Engine observation tests cover exact versions,
hybrid boundaries, eviction, and bounded ancestry. Router tests cover admission,
wait budgets, cancellation, retries, and stale snapshots.
