# Weights and checkpoints

## Publish before sampling

- `save_weights_and_get_sampling_client()`: latest-version pool, min-version guarantee (served at least the published version, possibly newer). Default for RL rollouts.
- `save_weights_for_sampler(name).result()` then `create_sampling_client(path)`: immutable named publication on a pinned exact-version pool (its own cold start). Use for eval and reproducibility.

## FFT deltas

Publications after the base are sparse deltas chained to preceding deltas; their paths cannot go to `load_state()`. Bigger updates write more bytes: lr ≥ 1e-5 produced unwieldy deltas in Qwen FFT RL runs; 1e-6 is the validated LR.

## LoRA bulletin

Publications write full adapter weights; one rollout pool serves all versions via per-container LRU eviction. Cache: `ROLLOUT_MAX_LOADED_LORAS`, `ROLLOUT_MAX_LORAS_PER_BATCH`. One publication in flight per adapter; parallel persist threads across clients. The backend accepts batches from older policies; retain the returned sampling logprobs.

## Checkpoints

- `save_state(name)` saves model + optimizer; `load_state_with_optimizer(path)` or `create_training_client_from_state_with_optimizer(path)` resumes.
- Volume layout `<name>/<model-id>/`; path form `tinker://<model-id>/weights/<name>`. No presigned archive URL; read with `modal volume get <volume> /<path> <dest>`.
- Publication is not a checkpoint: neither publish API saves optimizer state.
- Non-blocking: capture runs on the GPU lane, Volume persist overlaps later work. Keep ≤ 1 pending `save_state` future; await the final one before exit.
- Save on a schedule: preemption, container failure, or the 24h function timeout loses in-memory state.

## Recovery

- HTTP 410 `model_lost`: create a replacement in the same scope via `create_training_client_from_state_with_optimizer(path)` or a new client + `load_state_with_optimizer`. A second model is rejected while the first is healthy.
- After reassignment, old training and old latest clients get 410; base sampling and pinned versions survive. Resume the data iterator and step counter from the checkpoint position; steps after it must be rerun.
- Optimizer timeout may mean the update applied and only the response was lost; check the outcome before retrying.
- Backend transport failure terminates the process group; the command is not retried and the connection rejects further commands. Recover from a checkpoint; do not resubmit the failed command.
- Scope exit saves nothing; process retry creates a new scope; if the owner is killed, its apps stop automatically (~177 s observed).
- Diagnose sampling failures separately before replacing the trainer.
