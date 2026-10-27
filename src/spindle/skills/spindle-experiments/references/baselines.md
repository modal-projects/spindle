# Baselines

Async step time = one weight publication to the next (rollout fill + forward_backward + optim_step). Step 0 includes compile/cold start; compare medians.

## FFT

| Model × task | Setup | Result |
| --- | --- | --- |
| Qwen3.5-4B × DAPO Math | 30 steps, 8k cap | Spindle vs Miles reward plot (docs/validation.md) |
| Qwen3.5-9B × DAPO Math | 29 steps, async, 20k cap | comparable reward and step time vs Miles |
| Qwen3.6-27B × DAPO Math | 30 steps, 8k cap | Spindle vs Miles reward plot |
| Qwen3.6-35B-A3B × DAPO Math | 30 steps, 8k cap | run completed |
| Qwen3.5-9B × LongRLVR async | 30 steps, 8×H200 TP2 CP2 DP2, 16×8, 8k cap, 64k ctx | trainer 26–31% MFU after ~5-step compile ramp |
| Qwen3.5-9B × codeforces codegolf | 748 steps, async, 4×8, GRPO/TailRL | last-20 avg 95.2% correctness, ~40 s/step incl. publication; held-out 15/16 at step 740 (examples/codeforces-codegolf) |

## LoRA

| Model × task | Setup | Result |
| --- | --- | --- |
| Qwen3.5-9B-Base × GSM8K | 6 r16 clients × 30 updates, 4×H100 TP4 + 8 H200 | step 21.95 s (Miles 17.65), reward 0.659 (0.642) |
| Qwen3.5-9B-Base × DAPO Math | 6 r32 clients × 30, same hw, 8k cap | step 84.28 s (74.81), reward 0.156 (0.169), interval 43.27 min |
| Qwen3.5-9B × codegolf | 4 r32 clients × 500, 8×H200 TP8 + 4–8 H200, TailRL, 64k/16k | completed (docs/lora_validation.md) |
| Qwen3.5-9B × SWE-Gym | 8 r32 clients, async multi-turn, 128k ctx, 8×H200 + 8 H200, 32×8 | reward/step + logprob-diff plots |
| Qwen3.8-27B × LongRLVR 128k | r32, 4k cap, GRPO 16×8, lr 1e-4, 8×H200 TP2 CP4 + 4×TP2 | step 1676 s vs Miles 1658 s; reward 0.56 vs 0.57 (steps 0–8) |
| Qwen3.5-9B-Base × DPO HHH | lr 1e-4, 10 steps, stock 16k deploy | dpo_loss 0.6939→0.6795, acc 0.480→0.557, margin −0.0011→0.0502; ~21 s warm step + 9–14 s ref logprobs |

## Client scaling

One 8×H200 trainer + 8 single-H200 replicas, DAPO:

| Clients | Output TPS | $/M output tokens |
| ---: | ---: | ---: |
| 1 | 1,411 | $14.30 |
| 8 | 5,220 | $3.87 |
| 16 | 8,390 | $2.40 |
| 32 | 9,298 | $2.17 |

Per-client TPS falls (653 → 291 at 8 → 32); trainer busy-fraction ~84% at 8 clients.
