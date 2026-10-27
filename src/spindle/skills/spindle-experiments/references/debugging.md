# Debugging

## Startup

- HTTP 408 on sampling = replicas not ready; clears on its own.
- First `forward_backward` pays scheduling + boot + model load + compile.
- Find the dominant phase from `spindle_startup_mark` logs (command in monitoring.md). Fixes: `queue_s`/boot → capacity or `min_containers > 0`; model load → expected once; `first_forward_backward_done` → compile. A startup failure pauses new starts 5 min; `spindle deployment retry` clears it.
- Warmup failure on a scoped run can be an incomplete assets cache: an interrupted download leaves config/tokenizer but missing weight shards; verify all indexed files.

## Throughput

- Idle trainer is still billed. Causes: too few replicas (raise `max_containers`; keep `min_containers` warm for sync bursts), lockstep sync clients, not submitting futures before awaiting.
- Publication stalls: FFT delta size grows with LR; LoRA allows one in-flight publication per adapter. Check `spindle.trainer.capture.*`/`persist.*` spans.
- Profile a step: `SPINDLE_TORCH_PROFILE_STEP=2 uv run spindle deploy <config>` writes `rank<N>.trace.json.gz` to `/checkpoints/torch-profile/<definition id>`; open in Perfetto. `SPINDLE_TORCH_PROFILE_RANKS` (default `0`), ~250 MB/rank. Phases labelled `spindle/forward_backward`, `optim_step`, `forward_only`, `export_slot_peft`.
- Read the trace for idle GPU streams inside `spindle/forward_backward` (CPU/comms stalls), long `ncclKernel_*`, and time outside labelled phases (export, checkpoint writes).
- Under-populated tenancy is the common cost bug: $14.30 → $2.17/M output tokens at 1 → 32 clients on fixed compute.

## Numerics

- Train and sample kernels differ; small logprob drift is expected. Compare like the SWE-Gym mean/mean-abs logprob-diff plots before declaring a bug.
- Replay (Miles LoRA): request `return_sampling_mask=True`/`return_routed_experts=True`, call `spindle.sample_with_replay`, merge `sequence.replay.training_inputs(len(tokens))` into `loss_fn_inputs` after other entries, and pass `replay.loss_fn_config()` (preserves temperature). Mask is frozen from rollout. Router replay needs `num_layers` + `experts_per_token` on every datum, `ROLLOUT_RETURN_ROUTED_EXPERTS=True` on the pool, and `--use-rollout-routing-replay` in `backend_config["miles"]["extra_args"]`.
- Deterministic training: see the `spindle` skill's multi-lora.md. Known nondeterminism: FA3 backward kernel, unordered grad accumulation across packed microbatches.

## Recovery

Mechanics live in the `spindle` skill's weights.md (replacement clients, 410 handle invalidation, scope rules, timeout uncertainty). Diagnosis notes:

- Restores are not guaranteed bitwise: e2e restored forward logprobs within 1.2e-7 and the first loss exactly, but post-update logprobs differed up to 0.127 (unresolved, lr 5e-5 test).

## Errors

Control-plane mapping (`control_plane/http.py`):

| Error | HTTP | `error` | Action |
| --- | --- | --- | --- |
| `ModelLost` | 410 | `model_lost` | Replace client from checkpoint (weights.md) |
| `EngineSaturated` | 429 | `saturated` | Back off; check trainer slots/client fan-out |
| `SequenceConflict` | 409 | `conflict` | Regenerate seq ids; duplicate/resubmitted op |
| `RecordUnavailable` | 409 | `conflict` | Check object state (e.g. checkpoint metadata) |
| `RecordNotFound` | 404 | `not_found` | Verify model_id / paths |
| `RequestValidationError`/`ValueError` | 400 | `invalid_request` | Datum shapes, parameterization, `rollout` on non-FFT, non-`adamw` optimizer |
| `httpx.TransportError` | 503 | `engine_unreachable` | Retry; persistent = trainer down |
| unhandled | 500 | `internal` | Report `error_id` + server logs |
| `BackendFailed` | engine-internal | | Backend unusable; terminate + recover from checkpoint |

- `retrieve_future` pending → 408 `try_again` (`queue_state` `active`/`starting`); retryable-lost → 410 `request_lost` (resubmit).
- Second model in a scope → 400 `a training model is already active in this deployment`.
- 401 `unauthorized` → `TINKER_API_KEY` vs the `spindle-api` secret.
