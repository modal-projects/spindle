# LoRA-aware routing

Native Flash affinity remains the default (`inference_lora_routing = "affinity"`).
Set `inference_lora_routing = "cache_wait"` to enable the Rust LoRA-aware router
for exact-version LoRA sampling. Latest-version and base sampling retain native
routing because their exact adapter version is not known at selection time.

Configure `sglang_cfg.max_running_requests`, `sglang_cfg.max_queued_requests`,
and `inference_routing_work_rates`
with positive `prefill_tokens_per_second` and `decode_tokens_per_second`, and
nonnegative `gpu_lora_seconds` and `registration_seconds`. Calibrate these for
the deployment: the prefill rate controls waiting and distinguishes cheap initial
prefixes from substantial histories. The remaining rates provide diagnostic work
estimates, not selection scores.

## Selection

Among replicas with fresh observations, count active requests plus unobserved
local reservations. Capacity is the configured running limit plus the queued
request limit (zero when unspecified). Only replicas below capacity can receive a
request. Let `m` be the minimum count among available replicas. Exclude a candidate
when both `active - m > absolute` and `active > m * relative`.

A substantial cache match covers more than `inference_routing_cache_threshold`
of the prompt (default: 0.5) and saves at least 0.5 seconds of estimated prefill.
For these histories, use `inference_routing_balance_absolute` and
`inference_routing_balance_relative` (defaults: 4 and 1.25). For cheaper prefixes,
use `max(1, absolute // 2)` and `1 + (relative - 1) / 2`. This spreads new work
before long histories accumulate while allowing more imbalance to preserve
valuable existing history.

Pinned KV occupancy adds a second guard. Avoid a worker above 80% pinned occupancy
for cheap placement, or 90% for substantial histories, only when another
available worker within the request-count band is more than 15 percentage points
lower. Evictable cached blocks are not pinned work. Missing or invalid memory
observations leave request-count selection in effect.

Within these guards, prefer the longest observed prefix if it exceeds the cache
threshold. Cache ties and cold placements prefer a worker already serving more
requests for the exact adapter version, then fewer distinct active adapters.
Request count, adapter residency, group preference, and deterministic hashes
resolve remaining ties. Local reservations contribute to adapter counts without
inventing cache hits.

If a full, balanced replica has more cached tokens than any available replica,
retain its cache preference while every available alternative also has queued work.
The original request deadline bounds this wait. Once an alternative queue drains,
wait at most `min(2 seconds, extra_cached_tokens / prefill_tokens_per_second)`
before spilling to an available replica. The load-imbalance guard still applies. A completely full pool waits for
capacity without starting the affinity budget. Polling and target changes do not
reset an already-started budget. These estimates bound delay; they do not predict
end-to-end latency or guarantee a performance improvement for every workload.

A preferred adapter group can similarly wait when its worker is full or cooling
down and every available alternative already queues work. It must pass both load
guards and cannot displace a better available cache match. A drained alternative
queue permits spillover.

The current large SWE-Gym evaluation uses 128 running requests, eight queued
requests, an absolute balance allowance of 64, and a relative allowance of 2.
Its cheap-placement band is therefore 32 and 1.5. These are explicit experiment
settings; suitable values depend on the workload and engine capacity.

Samples sharing one request group may also wait up to two seconds for a dispatched
sibling's prefix to appear in an observation. The hint expires on leader completion,
failure, overload, or elapsed budget. It never counts as a confirmed cache hit.
The Python reference runtime shares this hint within a sampling group. The native
runtime currently prepares each dispatched sample independently; separate calls
do not share a pending-prefix hint.

## Prefix matching

The engine observer exports reusable device-cache boundaries scoped to the exact
LoRA identity and weight version. The Rust index stores compressed radix trees
over cumulative 128-bit fingerprints at 64-token boundaries. One traversal returns
the longest observed match on each replica. The final prompt token is excluded
because it must be evaluated for logits. Hashes are hints; SGLang performs the
actual cache lookup and version validation.

The structural catalog retains at most 8,192 boundaries per replica with complete
ancestry. A separate bounded endpoint catalog preserves reusable matches when
structural ancestors do not fit. The compact observation contains the union of
reusable endpoints from both catalogs; endpoint hashes are decoded once per
snapshot, outside request handling. Hybrid/Mamba boundaries without reusable
state are structural nodes only.
Evicted ancestors, host-only state, unsupported caches, and different adapter
versions receive no credit. Replacing snapshots removes evicted observations;
dispatching requests never manufactures cache entries.

The rollout image installs read-only cache export on the pinned SGLang fork.
At most once per second, the serving scheduler captures owned Rust tree rows,
adapter names, request counts and CPU pool counters. A background writer hashes
prefixes, selects the bounded catalog and atomically publishes the file. A
single-slot queue replaces an unpublished snapshot with the newest capture
without blocking scheduling. Mutable legacy Python trees are still inspected on
the scheduler thread.

Published timestamps retain the capture time, so publication delay cannot make an
old snapshot appear fresh. A failed publication preserves the previous file and
the writer continues with later captures. Native capture and Python binding
conversion still run on the scheduler; background Python work also shares the
GIL. Moving formatting does not by itself establish a GPU throughput improvement.
The sidecar exposes `/routing-state`; no benchmark endpoint or per-completion
archive is needed for routing.

## Execution and limits

Each sampling-worker process maintains one router per inference pool, shared across
adapters. There is no central request coordinator. The default Rust runtime owns
observation HTTP, prefix matching, admission, retries, and worker HTTP. Python
uses PyO3 only for startup and shutdown, publishes membership, and calls a
Bearer-authenticated loopback HTTP service through the common `RouterTransport`
interface. Snapshot construction runs off the native asynchronous executor.
Set `inference_router_runtime = "python"` to use the reference transport, which
calls the same scheduler through PyO3 for selection and reservation release.

Membership refreshes every five seconds and snapshots every half second.
Observations older than three seconds are ineligible. Reservations only cover the
local frontend; other frontends can race against the same worker snapshot.
SGLang remains responsible for final admission. HTTP 429/503 causes a short busy
cooldown without spending the eight-attempt transport/server-failure budget.
Busy workers remain visible for cache preference but cannot receive requests
during cooldown; failed workers are excluded. Transport failures cause a longer
health cooldown. Saturation and missing
observations wait until the request deadline, without native routing fallback.
Completion, failure, and cancellation release local reservations.

The sidecar shares one generation across repeated deliveries of the same scalar
request ID and payload. Reusing that ID with a different payload returns HTTP 409.
One disconnected caller does not cancel another caller waiting for the same
generation. When all callers disconnect, a one-second reconnect grace period
precedes cancellation; engine abort calls have a two-second deadline.

After an ambiguous connection failure, interrupted successful response, or proxy
408/5xx without worker identity, the native router makes one immediate retry to
the same worker with the same ID and payload. It retains its reservation and
original request deadline. Known worker busy/error responses use the normal
new-attempt path. A sidecar reply explicitly marking that attempt as cancelled
allows a fresh attempt; other HTTP 409 responses remain terminal. Exhausted
reconnects count toward the existing eight-attempt failure limit.

Completed replies are retained locally for at most 30 seconds, bounded by 512
entries and 64 MiB of response bodies. This does not provide exactly-once
execution across workers or restarts. Batch request IDs retain their existing
pass-through and per-request cancellation behavior.

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
