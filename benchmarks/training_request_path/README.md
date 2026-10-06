# Training request path: exact DAPO replay

These scripts replay saved inputs from the eight-client Qwen3.5-9B DAPO run of
2026-10-02. They do not generate a substitute dataset. Each client update contains
64 responses. Token IDs, rollout log probabilities, rewards and the original
advantage calculation are preserved. The CPU codec test uses updates 8–13; the
HTTP tests use updates 8–10.

## Changes

- Advertise the SDK's existing compressed protobuf upload support. Decode its
  tensor buffers directly, without converting them through JSON/base64.
- Move large request decoding and fingerprint construction off the HTTP event
  loop. Compute fingerprints before taking the engine state lock; admission and
  duplicate checks remain atomic inside that lock.
- Encode validated training arrays once. Reuse those bytes for the retry
  fingerprint and the private engine-to-backend request, replacing the large
  internal JSON encode/parse/validation pass. Preserve received float precision,
  sparse indices, shapes and multimodal input metadata.
- Run independent routing reads concurrently and share pending reads for the
  same model/instance. Share pending session heartbeat writes. Authorization is
  still checked for each request; completed reads are not cached.
- Separately, the included Tinker 0.24.1 patch counts tensor elements without
  allocating a Python list. It preserves the SDK's existing chunk boundaries.
  This patch is not installed by the Spindle changes.

The engine and backend must be updated together: the backend accepts both the
old JSON format and the new binary format, but an old backend cannot accept the
new sender. They run in the same trainer deployment. Public JSON clients remain
supported. No result format, loss function or optimizer behavior changes.

## Correctness

The HTTP benchmark computes SHA-256 over an independent canonical JSON
representation before submission and again at executor entry. It compares the
complete multiset of `(model_id, payload_hash)` pairs, checking for altered,
missing and duplicate chunks. Every measured arm delivered all 217 chunks from
24 client updates (24,772,945 input tokens) exactly. The longer CPU codec replay
also matched all 48 updates: 3,072 datums and 49,331,231 input tokens.

244 engine, control-plane, provider and replay tests pass. They include exact
numeric preservation, sparse and multimodal data, JSON/protobuf retry equivalence,
changed-data rejection, compressed-request rejection with sequence advancement,
concurrent routing, closed sessions and cancelled liveness waiters. Two separate
SDK tests verify unchanged chunk-size estimates without tensor-list conversion.

This establishes data preservation through executor entry. A real GPU A/B is
still needed to compare losses, parameter updates and actual training latency.

## Measurement boundaries

`http_replay.py` runs the real SDK, frontend, engine and backend as separate local
HTTP processes. Eight clients run concurrently; arms run sequentially and each
arm's processes are terminated before the next starts. One update per client is
excluded as warmup, leaving 16 measured calls per arm.

- **Backend arrival**: immediately before the backend audit hashes its inputs,
  measured from the client calling `forward_backward()`.
- **Forward result**: until the client receives the executor result. This includes
  the audit and SDK result polling.
- The executor is a CPU stub returning small synthetic results. There is no GPU
  computation, realistic per-token result payload, Modal network or remote KV
  latency in these measurements. They cannot establish training TPS.
- `cpu_replay.py` isolates encoding/decoding on CPU, excluding HTTP, queues,
  routing, GPU execution, result delivery and SDK chunk sizing. It encodes a full
  client batch rather than SDK-sized chunks.

The isolated 8×H200 baseline deployment remained queued with zero runners for
approximately 40 minutes. It never executed a training step. All isolated test
apps were then stopped and verified to have zero tasks. Existing deployments
were not redeployed. The GPU comparison is blocked on allocation; no conclusion
about actual training throughput or model-update equivalence is claimed.

## Local HTTP results (seconds)

![Local HTTP latency; no GPU computation](request-latency.png)

| Code / SDK | Mean backend arrival | Mean forward result |
| --- | ---: | ---: |
| Main + released SDK, run 1 | 48.52 | 55.94 |
| Main + released SDK, run 2 | 48.46 | 58.19 |
| Optimized server + released SDK | 8.54 | 14.08 |
| Optimized server + sizing patch | 8.70 | 17.55 |

The server changes cut mean backend-arrival latency by about **82%** in this
local replay. The SDK patch has no demonstrated incremental latency benefit in
these runs; asynchronous batching and result polling add variability. Two earlier
optimized runs, before the final vectorized protobuf decoder, measured 8.84–9.12 s
backend arrival and 15.00–16.97 s forward-result latency. All arms matched their
expected data hashes.

The preliminary CPU-only codec replay reduced mean encode/decode work per full
client batch from 6.72 s to 0.95 s. Mean upload size fell from 41.1 MB JSON to
5.49 MB compressed protobuf; internal payload size fell from 45.2 MB JSON to
32.9 MB binary. These are decimal MB and are distinct from the HTTP measurements.

## Reproduce

Use an environment with Spindle's test dependencies and Tinker 0.24.1. Set
`ARCHIVE` to the saved run's `training` directory. Keep result files outside the
repository. Do not run timed arms simultaneously.

```bash
PYTHONPATH=src:. python benchmarks/training_request_path/http_replay.py \
  --archive "$ARCHIVE" --output /tmp/request-path-after
python benchmarks/training_request_path/summarize.py \
  --clients /tmp/request-path-after/clients \
  --backend /tmp/request-path-after/backend.log \
  --output /tmp/request-path-after-summary.json
```

For the baseline, use Spindle commit
`7279288e095d7d1b42cd6abb902cbf87ddc6039d` on `PYTHONPATH` with the same harness and
stock SDK. For the optional SDK improvement, apply
`sdk/tinker-0.24.1-sizing.patch` to Tinker commit
`b9dbbef6f9967cb7d6666fb1b79a2dba12604c5e` and put that source on `PYTHONPATH` too.

`prepare.py before|after DESTINATION` creates isolated deployment source copies
with equal CPU allocation, broad US trainer placement, compilation disabled in
Ray workers, and the same executor audit. `deployment.py` describes the GPU
configuration; change the test deployment names and storage names for a new run.
Deployment is explicit, never performed by the replay scripts. For a GPU run,
use `replay.py --archive "$ARCHIVE" --base-url URL --output OUTPUT` (six updates)
and summarize with `--warmup 2`. Stop the first arm completely before starting
the second. Stop all isolated apps afterwards, including on client failure.
