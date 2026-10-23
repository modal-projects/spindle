# LoRA inference routing

Modal affinity remains the default. Set `inference_lora_routing = "cache_wait"`
to route exact-version LoRA samples using observed KV reuse, adapter locality,
request load, and pinned KV-memory pressure. The Rust runtime owns polling,
admission, retries, and worker HTTP. Python starts the runtime and submits requests.

The load thresholds and estimated service rates belong in your deployment recipe.
For reference, the larger SWE-Gym experiments used the following settings; the
rates are measurements from that hardware and should be recalibrated for a
different model or GPU layout:

```python
inference_lora_routing = "cache_wait"
inference_routing_balance_absolute = 64
inference_routing_balance_relative = 2.0
inference_routing_cache_threshold = 0.5
inference_routing_work_rates = {
    "prefill_tokens_per_second": 10463.96,
    "decode_tokens_per_second": 1290.0,
    "gpu_lora_seconds": 0.075,
    "registration_seconds": 0.68,
}
```

Those experiments also used `max_running_requests = 128` and
`max_queued_requests = 8` in `sglang_cfg`, with
`inference_target_concurrency = 128`. These are workload-specific settings,
not new defaults. Enabling the router alone retains the default load thresholds
of 4 requests and 1.25×, which can spill requests much earlier.

The router excludes observations at least three seconds old, failed workers,
and exhausted local admission capacity. It favors useful cache matches within
request-count and pinned-memory guards. A worker exceeds the request guard only
when both the absolute difference and relative ratio exceed their thresholds;
short prefixes get a tighter placement band than substantial histories.
New placement uses an 80% pinned-memory threshold, while valuable histories can
use 90%, with a 15-percentage-point imbalance required for either to spill.
Evictable cache blocks count as reclaimable capacity, not pinned load.

Sibling samples share a bounded pending-prefix hint for their exact prompt and
adapter version; this hint never counts as an observed hit. Cache waiting is
normally bounded by estimated prefill savings and two seconds. When every
alternative is already queued, locality may be retained until the request
deadline. Reservations are local to each frontend; SGLang enforces final admission.

The engine captures residency metadata and changed token segments on its
scheduler thread. A separate CPU process reuses prefix hashes and publishes a
bounded set of reusable endpoints. Only one acknowledged delta is in flight;
a stalled or failed publisher triggers a full resynchronization. Observations
retain their capture time through publication and polling. No cache match,
eviction, access counter, or GPU tensor is modified by observation.

The observer supports the pinned Rust unified cache and legacy radix/Mamba
caches with data and pipeline parallelism both equal to one. Unsupported cache
layouts publish no hints; the router waits for valid observations rather than
silently changing routing policies.
