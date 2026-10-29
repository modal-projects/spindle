---
name: spindle-experiments
description: Run or diagnose a training experiment on a Spindle server (RL, SFT, DPO via the Tinker SDK).
when_to_use: Launch, continue, evaluate, monitor, or debug a Spindle training run. Deployment mechanics live in the `spindle` skill.
---

# Spindle experiments

## Scope

- Asked to train: own plan → prove → smoke → monitor → promote. Asked for one stage: only that stage. A status request is not a relaunch; a diagnosis request is not a fix.
- Compute-billed: cold starts and idle capacity are real spend in the user's workspace.
- For Tinker SDK usage, datums/renderers, losses, hyperparameters, RL/SFT/DPO/distillation methodology, and generic debugging, use tinker-cookbook's skills (install in Claude Code: `/plugin marketplace add thinking-machines-lab/tinker-cookbook`, then the `tinker` plugin: `tinker:research`, `tinker:debug`), pointing `base_url` at the Spindle server.
- Locate installed Spindle source: `python -c "import spindle.control_plane.http as m; print(m.__file__)"`.

## 1. Plan

Confirm with the user before writing the loop:

- Model + bundled config: smallest whose `max_context_length` fits the longest prompt + completion. List from `spindle.configs` or `/api/v1/spindle/deployments`.
- LoRA vs FFT: LoRA shares one engine across adapters; FFT gets a dedicated trainer and publishes deltas sized by the update.
- Eval metric, run on the base model first, on a pinned publication.
- Hyperparams from a validated run's row in [training.md](references/training.md), not defaults.
- Horizon and cost: `GPU $ = allocated GPUs × seconds`; estimate from smoke-step TPS, trainer and inference separately.
- Present proof, smoke, and full run as separate approvals.

## 2. Prove one step

sample → `forward_backward` → `optim_step` → `save_weights_and_get_sampling_client` → sample again (the README script). Advance only on nonempty tokens + logprobs, finite loss, clean publish. Time cold start separately; stalls go to [debugging.md](references/debugging.md#startup).

## 3. Smoke (~10 steps)

Same config, real batch size. Compare step timing (rollout / train / publish) and reward against the closest run in [docs/validation.md](https://github.com/modal-projects/spindle/blob/main/docs/validation.md) or [docs/lora_validation.md](https://github.com/modal-projects/spindle/blob/main/docs/lora_validation.md). One change at a time; consult the symptom table before changing anything.

## 4. Promote and report

- Monitor step time, reward, response length, publication latency, queue depth: [monitoring.md](references/monitoring.md).
- Early-stop on sustained no-progress across comparable evidence, not one noisy point.
- Scheduled non-blocking `save_state`; await the final save.
- Report run/app ids, checkpoint path, early-vs-late reward, eval, step time, GPU-hours. Verify `modal app list` is clean; stop leftover `spindle-fft-...` samplers.

## Diagnose by symptom

| Symptom | [debugging.md](references/debugging.md) section |
| --- | --- |
| 408s, hangs, slow cold start | startup |
| Slow steps, idle trainer, rollout bottleneck | throughput |
| Where a step's GPU time goes | [profiling.md](references/profiling.md) |
| logprob drift, nondeterminism | numerics |
| 410 `model_lost`, uncertain updates, retries | recovery |
| Unrecognized API error | errors |

Spindle-specific SDK notes, loss support pointers, staleness, DPO, validated hyperparams: [training.md](references/training.md).
