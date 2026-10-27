# Training settings

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
- LoRA LR: 1e-6 (RL e2e) and 1e-4 (128k LongRLVR, DPO) both validated. No repo evidence for a LoRA-vs-FFT ratio.
- Batches: 4–32 groups × 8; overfetch groups (DAPO script uses 4×) to drop degenerate ones.
- Rank 16 (GSM8K) or 32 (rest). Max rank per deployment (`max_lora_rank`); alpha/target modules deployment-side; SDK defaults `train_attn`/`train_mlp`/`train_unembed` true (wandb example: `train_unembed=False`).
- No validated run uses KL or an entropy bonus.

## Advantages

```python
advantage = (reward - group_mean) / max(group_std, 0.5)   # codegolf
advantage = centered / (centered.std() + 1e-6)            # Miles GRPO parity, dapo script
```

## Loss

`importance_sampling` and `ppo` take identical inputs; `ppo` adds clip thresholds. Validated configs are one of the two.

## Staleness

- `max_steps_off_policy = 1` in e2e scripts; codegolf discards batches > 4 weight versions old (recorded lag can overestimate real lag).
- Retain per-sequence sampling logprobs; the backend accepts older-policy batches but the loss needs behavior logprobs.
- Cancelling local rollout workers does not guarantee their Modal calls stopped.

## Degenerate groups

Constant-reward groups (all-fail/all-success) give zero advantage and consume the batch; drop them (`scripts/run_dapo_math_spindle_full.py --remove-constant-reward-groups`).

## Watch

Reward trend, response length (128k LongRLVR: Spindle's stayed longer than Miles', a loss-weighting delta under investigation), entropy/diversity (codegolf: mean negative sampled-token logprob, exact-string distinct fraction), truncation, policy lag/discard rate (codegolf two-batch queue 30.4% vs four-batch 43.5%).

## SFT / DPO

- SFT: `cross_entropy` + `weights` mask on prompt positions; no dedicated repo recipe.
- DPO: the tinker-cookbook recipe runs unmodified on the stock `qwen35-9b-lora-16k` deployment (`base_url`, `model_name=Qwen/Qwen3.5-9B-Base`, `dataset=hhh`, `renderer_name=role_colon`, `learning_rate=1e-4`, `dpo_beta=0.1`, `max_steps=10`). It uses `forward_backward_custom` (forward + `cross_entropy` with per-token weights); reference logprobs come from `compute_logprobs` on a sampler published from step-0 weights, adding 9–14 s to ~21 s warm steps. Pin that sampler; the latest pool would drift. Validated: dpo_loss 0.6939 → 0.6795, accuracy 0.480 → 0.557, margin −0.0011 → 0.0502; hosted Tinker tracked the same trajectory (0.6837).
