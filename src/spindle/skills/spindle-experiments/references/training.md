# SDK and training settings

## Spindle-specific SDK notes

- FFT client: `spindle.create_full_training_client(service, model)` (+ `_async` variant). `rollout`/`user_metadata` semantics: the `spindle` skill's engines.md.
- Replay: `spindle.sample_with_replay` returns replay tensors for `loss_fn_inputs` (Miles LoRA only; Megatron rejects replay fields).
- `target_tokens` accepts `[seq, choices]` on Miles only; weights/advantages/logprobs must match that shape.
- Megatron derives missing `weights` from positions with nonzero logprob or advantage.
- Loss support: `LossFn` in `spindle/backends/contract.py` (all names), `SUPPORTED_LOSSES` in `spindle/backends/miles_runtime/data.py` (Miles subset), `loss_fn_config` defaults in `spindle/backends/megatron_runtime/common/forward_backward.py`. Miles and Megatron support different sets; check both before choosing.
- Versions: `tinker>=0.24.1`. Deploy the current server before upgrading clients: 0.25 needs protobuf responses, 0.26 needs per-sequence sample ids.

## Validated runs

| Run | Param | LR | Loss / config | Groups × size |
| --- | --- | --- | --- | --- |
| DAPO Math comparison, Qwen3.5-9B, 30 steps, 32k ctx | LoRA r32 | 1e-6 | ppo 0.8/1.28, grad clip 1.0, temp 1.0 | 32 × 8 |
| LongRLVR e2e, Qwen3.5-35B-A3B | LoRA | 1e-6 | grpo | 16 × 8 |
| LongRLVR 128k, Qwen3.8-27B, 4k cap | LoRA r32 | 1e-4 | ppo 0.8/1.28, no KL | 16 × 8 |
| Codeforces codegolf, Qwen3.5-9B, 64k ctx | FFT | 1e-6 | ppo 0.8/1.2, temp 1, no KL/entropy bonus | 4 × 8 |
| DPO HHH, Qwen3.5-9B-Base, 10 steps | LoRA | 1e-4 | dpo_beta 0.1 | n/a |
| Single-step / wandb example | LoRA | 1e-5 | importance_sampling | 4 |

- FFT LR: all validation runs 1e-6; ≥ 1e-5 enlarges deltas (weights.md in `spindle` skill).
- LoRA LR: 1e-6 (RL e2e) and 1e-4 (128k LongRLVR, DPO) both validated; no repo evidence for a LoRA-vs-FFT ratio.
- Rank 16 (GSM8K) or 32 (rest). Max rank per deployment (`max_lora_rank`); alpha/targets deployment-side; SDK defaults `train_attn`/`train_mlp`/`train_unembed` true (wandb example: `train_unembed=False`).
- No validated run uses KL or an entropy bonus.
- Batches 4–32 groups × 8; overfetch (DAPO script uses 4×) and drop constant-reward groups (`scripts/run_dapo_math_spindle_full.py --remove-constant-reward-groups`).

## Staleness

- `max_steps_off_policy = 1` in e2e scripts; codegolf discards batches > 4 weight versions old (recorded lag can overestimate real lag).
- Retain per-sequence sampling logprobs; the backend accepts older-policy batches but the loss needs behavior logprobs.
- Cancelling local rollout workers does not guarantee their Modal calls stopped.

## Watch

Reward trend, response length, entropy/diversity (codegolf: mean negative sampled-token logprob, exact-string distinct fraction), truncation, policy lag/discard rate.

## DPO

The tinker-cookbook DPO recipe runs unmodified on the stock `qwen35-9b-lora-16k` deployment (`base_url`, `model_name=Qwen/Qwen3.5-9B-Base`, `dataset=hhh`, `renderer_name=role_colon`, `learning_rate=1e-4`, `dpo_beta=0.1`, `max_steps=10`). Reference logprobs come from `compute_logprobs` on a sampler published from step-0 weights; pin that sampler since the latest pool would drift.
