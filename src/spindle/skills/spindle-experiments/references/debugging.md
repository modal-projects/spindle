# Debugging

## Startup

- HTTP 408 on sampling = replicas not ready; clears on its own.
- First `forward_backward` pays scheduling + boot + model load + compile.
- Find the dominant phase from `spindle_startup_mark` logs (command in monitoring.md). Fixes: `queue_s`/boot → capacity or `min_containers > 0`; model load → expected once; `first_forward_backward_done` → compile. A startup failure pauses new starts 5 min; `spindle deployment retry` clears it.
- Warmup failure on a scoped run can be an incomplete assets cache: an interrupted download leaves config/tokenizer but missing weight shards; verify all indexed files.

## Throughput

- Idle trainer is still billed. Causes: too few replicas (raise `max_containers`; keep `min_containers` warm for sync bursts), lockstep sync clients, not submitting futures before awaiting.
- Publication stalls: FFT delta size grows with LR; LoRA allows one in-flight publication per adapter. Check `spindle.trainer.capture.*`/`persist.*` spans.
- Profile where a step's GPU time goes: [profiling.md](profiling.md).
- Under-populated tenancy is the common cost bug; see the client-scaling table in [docs/multi-lora.md](https://github.com/modal-projects/spindle/blob/main/docs/multi-lora.md).

## Numerics

- Train and sample kernels differ; small logprob drift is expected. Compare like the SWE-Gym mean/mean-abs logprob-diff plots before declaring a bug.
- Replay (Miles LoRA): request `return_sampling_mask=True`/`return_routed_experts=True`, call `spindle.sample_with_replay`, merge `sequence.replay.training_inputs(len(tokens))` into `loss_fn_inputs` after other entries, and pass `replay.loss_fn_config()` (preserves temperature). Mask is frozen from rollout. Router replay needs `num_layers` + `experts_per_token` on every datum, `ROLLOUT_RETURN_ROUTED_EXPERTS=True` on the pool, and `--use-rollout-routing-replay` in `backend_config["miles"]["extra_args"]`.
- Deterministic training: see the `spindle` skill's multi-lora.md. Known nondeterminism: FA3 backward kernel, unordered grad accumulation across packed microbatches.

## Recovery

Mechanics live in the `spindle` skill's weights.md (replacement clients, `model_lost` handle invalidation, scope rules, timeout uncertainty).

## Errors

Status/error-string mapping: `ERROR_STATUSES` in `spindle/control_plane/http.py`; engine-level handlers in `spindle/engine/http.py`.

- `ModelLost` (`model_lost`): replace the client from a checkpoint (weights.md in `spindle` skill).
- `EngineSaturated` (`saturated`): back off; check trainer slots and client fan-out.
- `SequenceConflict` (`conflict`): regenerate seq ids; duplicated or resubmitted operation.
- `RecordUnavailable` (`conflict`): check object state, e.g. checkpoint metadata.
- `RecordNotFound` (`not_found`): verify model_id and paths.
- `invalid_request`: datum shapes, parameterization, `rollout` on a non-FFT client, or a non-`adamw` optimizer.
- `engine_unreachable`: retry; persistent means the trainer is down.
- `internal`: report the `error_id` with server logs.
- `BackendFailed`: backend unusable; terminate and recover from a checkpoint.
- `retrieve_future` pending returns `try_again` with `queue_state` `active`/`starting`; a retryable lost future returns `request_lost`, resubmit the operation.
- A second model in a scope is rejected with `a training model is already active in this deployment`.
- `unauthorized`: `TINKER_API_KEY` does not match the `spindle-api` secret.
