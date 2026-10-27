# SDK

## Clients and calls

- `tinker.ServiceClient(base_url, api_key)`
- LoRA: `service.create_lora_training_client(base_model=..., rank=...)`
- FFT: `spindle.create_full_training_client(service, model)` (`_async` variant exists; `rollout`/`user_metadata` semantics in the `spindle` skill's engines.md)
- Sampling: `save_weights_and_get_sampling_client()`, `save_weights_for_sampler(name)` → `create_sampling_client(path)`, `compute_logprobs` for reference logprobs, `spindle.sample_with_replay` for replay tensors (Miles LoRA only)
- Checkpoints: `save_state(name)`, `create_training_client_from_state_with_optimizer(path)`, `load_state_with_optimizer(path)`
- `forward_backward`/`optim_step` return futures: submit both before awaiting so they pipeline (the engine buffers the next command). Async: `*_async` + `result_async` + `asyncio.gather`.
- Versions: `tinker>=0.24.1`. Deploy the current server before upgrading clients: 0.25 needs protobuf responses, 0.26 needs per-sequence sample ids.

## Datums

Shift-by-one pairs, all fields equal to `model_input` length:

```python
model_input = prompt + answer[:-1]
target_tokens = prompt[1:] + answer
```

Mask prompt positions with `weights` (cross_entropy) or zeros in `logprobs`/`advantages` (RL). Megatron derives missing `weights` from positions with nonzero logprob or advantage. Miles requires `target_tokens` plus the loss-specific inputs below. Miles accepts `target_tokens` shaped `[seq, choices]`; weights/advantages/logprobs match that shape.

## Loss functions

| `loss_fn` | `loss_fn_inputs` beyond `target_tokens` | `loss_fn_config` defaults | Backends |
| --- | --- | --- | --- |
| `cross_entropy` | optional `weights` (1.0) | none | all |
| `importance_sampling` | `logprobs`, `advantages` | none | all |
| `ppo` | same | `clip_low_threshold` 0.8, `clip_high_threshold` 1.2 | all |
| `cispo` | same | 0 / 4 | all |
| `dro` | same | `beta` 0.05 | all |
| `dppo` | same | `tv_threshold` 0.1 | Megatron only |

Miles replay fields (`sampling_mask_*`, `routed_experts`) only via `sample_with_replay`; Megatron rejects them. Batches group only when `loss_fn` and `loss_fn_config` match.
