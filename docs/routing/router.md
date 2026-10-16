# Inference routing

`sample_task` prepares one request per sample group. `sample_one` calls
`await router.select_destination(request, attempt_id, deadline=...)`, sends to
that destination through Flash, and calls `router.complete(...)` in `finally`.
Sampling does not inspect replica catalogs, scores, or reservations.

- `rust/router/src/policy.rs`: interchangeable policies. `joint` compares
  outstanding prefill/decode work plus uncached prefill and adapter setup, all
  in calibrated seconds. `least_load` uses outstanding request count.
- `rust/router/src/lib.rs`: version-scoped prefix hashes, observations, capacity
  eligibility, group placement, reservations and outcome feedback. Siblings
  reuse their selected host while it has capacity; they can spill when full.
- `src/spindle/inference/router.py`: background refresh, bounded asynchronous
  capacity waits, and shutdown. No scheduling formulas.
- `src/spindle/providers/modal/router.py`: Modal discovery and replica addresses.

Choose `inference_lora_routing = "joint"` or `"least_load"` in a recipe. The
unchanged default, `"affinity"`, uses native Flash routing. The Rust policies
require `sglang_cfg.max_running_requests` and `inference_routing_work_rates`
(prefill/decode tokens per second, GPU-load seconds, registration seconds).
Both use identical observations, admission and grouping for policy comparisons.

One router lives in each sampling-worker process per inference pool, shared
across that pool's adapters. No central coordinator. Snapshot replacement and
selection are short synchronous Rust calls under Python's normal interpreter
lock; network operations happen outside them. This implementation does not
claim lock-free parallel selection within one process.

Discovery runs in the background every 0.5 seconds; container membership is
refreshed every 5 seconds. Observations older than 3 seconds are ineligible.
Reservations account for local requests not yet visible in an observation.
Other workers' unseen requests can still race; SGLang owns final admission.
503/429 means busy, not unhealthy. Saturation waits locally and never turns
into native routing fallback. Request deadlines and cancellation still apply.

KV identity includes the exact adapter version and 64-token prefix boundaries.
The last input token is excluded from reusable credit. A version change never
inherits cache credit from an earlier version. The estimated remaining decode
work uses the output budget; natural EOS makes it an estimate, not a latency
prediction. Latest-version sampling retains the native path because its exact
version is resolved downstream.

Build locally with `maturin build --release --manifest-path rust/router/Cargo.toml`
and install the wheel. Installing Spindle from source also builds the extension
through setuptools-rust. Modal builds the wheel in its frontend image; GPU
engine images do not need this router extension.

Validate with `cargo test --manifest-path rust/router/Cargo.toml` and
`pytest tests/inference/test_rust_router.py tests/inference/test_sampling.py`.
Set `PYO3_PYTHON` to a Python 3.12 interpreter when building Rust tests.
