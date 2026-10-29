# Add a backend

- Implement `Backend` in `backends/contract.py` (`accept_model`, `forward_backward(ForwardBatch)`, optimizer step, `load_checkpoint`, capture/persist/publish split, `unload_model`, shutdown) plus `build_executor()` returning the `DistributedExecutor`. A model definition passes `module:build_executor` to `run_engine_with_backend`.
- Rank 0 exposes the HTTP bridge; followers run `run_follower_loop` (engine/spmd.py). `HttpBackendClient` crosses the subprocess boundary.
- Lock-free lanes: `capture_*` must return detached CPU state the persist call exclusively owns. References `backends/megatron_lora.py`, `backends/megatron_fft.py`.
- A new loss needs the backend's supported set + `loss_fn_config` validation; contract `LossFn` is the union.
- Tests: `tests/backends`, `tests/engine` (CPU). GPU smoke pattern: `tests/gpu` + `scripts/check_multi_targets_gpu.py` (temporary H100s, no deploy).
