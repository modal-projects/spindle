# Asynchronous mixed-sampler reproduction

This reproduces the training setup described in Modal's
[mixed-precision sampler article](https://modal.com/blog/toward-rl-on-mixed-precision-mixed-accelerator-samplers)
using native Spindle `score_centered_mis`. The implementation follows the
[score-centering estimator](https://arxiv.org/html/2609.20807), including its
correction when MIS masks the sampled token.

**Training stability has not yet been established.** Unit tests and short GPU
checks validate implementation paths; the asynchronous experiment must produce
training curves before a stability conclusion is justified.

## Settings

- `Qwen/Qwen3.6-35B-A3B`, pinned to `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- `XiaomiMiMo/MiMo-V2.6-RL-oss`, code subset, pinned to the article-era revision
  `639865fd3374018d6cb29b9fb82dd531406fcf5f` containing **2,698 tasks**. The later
  release removed 96. Images come from this revision's `image-mapping.jsonl`.
- BF16 full-parameter trainer on eight B200s; 262,144-token context; batches of
  128 prompts with eight trajectories per prompt; no MoE route replay.
- Eight sampler groups, all with FP8 KV cache:

| Group | GPUs per replica | Weight format |
|---|---:|---|
| B200 | 2 | NVFP4 |
| B300 | 2 | NVFP4 |
| H100 | 4 | FP8 |
| H100 | 4 | BF16 |
| H200 | 2 | FP8 |
| H200 | 2 | BF16 |
| A100 80GB | 4 | BF16 |
| RTX PRO 6000 | 4 | BF16 |

Each trajectory stays on one group. Initial measured throughput determines fixed
assignment probabilities. The replica counts and GPU counts per replica are
choices for this reproduction; the article does not specify those allocations.

Rollout producers continue during training, optimizer updates and publication.
The trainer consumes completed prompt groups, and never rejects a group based
on its policy age. Bounded queues limit outstanding work. Event timestamps record
both overlap and actual served weight versions, including updates within a turn.

## Explicit assumptions

The original private code and run configuration are unavailable. The article
does not specify learning rate, top-p threshold, optimizer details, exact agent
prompt, or the full sanitization procedure. This recipe chooses:

- top-p 0.9, temperature 1, top-k disabled;
- MIS ratio window [0.2, 5.0]; per-prompt reward normalization with population
  standard deviation plus 1e-6; token-mean loss over the full update;
- Adam, LR 1e-6, betas (0.9, 0.95), epsilon 1e-8, no weight decay, clip norm 1;
- up to 200 bash turns; each continuation may use the remaining 256K context budget;
- a three-hour sampling request deadline, allowing long continuations to finish;
- stop after three consecutive non-tool responses; the `submit` tool
  ends the agent session; binary test reward;
- 192 concurrent prompt producers, at most 512 CPU rollout workers, and up to
  two serving replicas per group, keeping one warm through fleet startup;
  target of 200 updates, checkpoint after every optimizer update.

All groups use PyTorch top-p selection so recorded supports and probabilities
describe the actual categorical distribution. Their model execution still uses
their accelerator-specific and quantization-specific kernels. Exact support
capture is bounded by the full model vocabulary, not an approximate top-k tail.

The public `XiaomiMiMo/mimoagent` environment is pinned to
`c3df8469f250b15e7906487a6d52be4c1827a509`. Setup runs its anti-leak cleanup,
then replaces repository history with a single root commit. Grading applies the
submitted diff in a fresh sandbox and runs the hidden test patch there. Verifier
infrastructure errors fail the rollout rather than becoming zero rewards.
Container tags are published dataset tags; historical image digests are not
available, so this is not a bitwise reconstruction of the original environments.

NVFP4 uses the pinned SGLang runtime's `nvfp4_online` routed-expert quantization;
dense layers retain their source precision. Staging recreates NVFP4 loader
callbacks against the isolated CPU layer, keeping its weights and scales
together and preventing writes to live serving scales during compilation.
Decode-scale arithmetic matches GPU startup during CPU staging as well.
FP8 quantizes BF16 weights into E4M3
with FP32 scales per 128 x 128 block, matching the granularity of
[Qwen's official FP8 configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/main/config.json).
Recurrent gates and routers retain their source precision. This is recomputed
for every BF16 publication; there is no separately trained FP8 base model.
The serving weight cache recompiles BF16 publications into each group's format.
`validate --weights` first checks compiled runtime weights byte for byte against
startup weights, then tests a changed expert tensor followed by restoration using
the same XOR-delta format as Spindle publications. Exact sampled text equality is
recorded separately because the normal serving kernels are not deterministic.

## Run in kailash-dev

From this directory, using Python 3.12:

```bash
uv sync --frozen
export MODAL_ENVIRONMENT=kailash-dev
uv run --frozen spindle deploy mixed_samplers/config.py --refresh-trainer qwen36-mimo-mixed-256k
uv run --frozen python -m mixed_samplers.validate --group all --weights --output sampler-validation.json
uv run --frozen python -m mixed_samplers.validate_trainer --api-key-file /path/to/deployment-key
uv run --frozen python -m mixed_samplers.train --api-key-file /path/to/deployment-key --output results/run-001
```

The deployment uses its own frontend, storage volumes and API-secret reference
`spindle-mimo-mixed-api`. The secret contains `TINKER_API_KEY`; the key file must
match it. Without `--api-key-file`, the clients use `TINKER_API_KEY` from their
environment. The existing `spindle-proxy` secret remains inside Modal workers.
The trainer has no geographic restriction; its API frontend remains in
`us-west`. This avoids requiring an idle eight-B200 node in one region.
H100 groups use Modal's `H100!` selector to prevent automatic H200 upgrades.
The explicit trainer refresh publishes native-code changes even when the trainer
configuration has not changed; an ordinary redeploy can retain the prior trainer
code. Check the running backend's diagnostic aggregation in the event log.

`validate_trainer` performs a synthetic native-loss check and unloads that model.
It does not contribute samples or optimizer state to the experiment. The training
controller owns ephemeral sampler/worker apps: keep it alive for the duration
of the run. A failed controller stops those apps. A trajectory gets at most one
infrastructure retry; repeated failure, non-finite metrics, a skipped optimizer
step, or two hours without a completed prompt stops the run.
Backward and optimizer waits are bounded at two hours, publication and checkpoint
waits at one hour. The controller requests trainer unloading when it exits.

Resume with `--checkpoint <path> --start-step <saved-step> --steps <remaining>`
using `checkpoint.json`. Optimizer state is restored. Queued rollouts and producer
RNG state are not resumed, so recovery is not a bitwise replay of scheduling.
Use a new output directory for each resumed attempt.

If a trainer is lost before its first optimizer update, restart with
`--initial-rollout-cache <previous-run-directory>` and a new output directory.
This checks recipe pins, sampling settings, task order and served versions before
reusing completed pretrained-policy trajectories. Missing trajectories are sampled
by the independent producers. Restored trajectories have distinct events, so they
do not count as newly completed rollouts in the overlap report. This option cannot
be combined with checkpoint restoration or a nonzero starting step.
Repeated recovery also retains trajectories restored by earlier attempts. Pending
gradients are not checkpointed, so an interrupted update must be recomputed.
Recovery retains the original validated capacity weights and runs fresh sampler
checks alongside training, recording them in `startup-validation.json`. This lets
the trainer consume saved trajectories while new serving replicas initialize.
A sampler check failure cancels training. A100 compute can use any US region to
avoid a capacity bottleneck; its request endpoint remains in `us-west`.

## Evidence to inspect

Native diagnostic totals preserve token averages when Tinker splits a large
request into chunks. The recipe reconstructs entropy, KL and MIS-mask means from
those totals. Older deployed backends expose only means; reports explicitly label
that fallback `sdk_chunk_means`. This affects diagnostic weighting, not the loss
gradient, whose active-token weights are normalized across the full update.

`manifest.json` records revisions, allocations, capacity measurements and chosen
settings. `events.jsonl` records per-trajectory rewards, formatting failures,
lengths and served versions; update records include entropy, sampler/trainer KL,
MIS rejection fraction, gradient norm, and successful updates. Compressed
trajectory files preserve the token IDs and full sampler distributions.

Review reward, format-error rate, entropy, length, KL and gradient norm over the
entire run, broken down by sampler group. Finite metrics alone do not establish
stable dynamics: check for reward collapse, degeneration, or trends that worsen
with policy age. The homogeneous B200 BF16 baseline and SWE-bench Pro V2/HARD51
evaluations from the article have not been reproduced by this script.

Local tests: `uv run --frozen pytest tests`. Native gradient and SDK tests live
in the repository's `tests/backends/test_score_centering.py` and `tests/replay`.
Generate measured curves with `uv run --frozen python -m mixed_samplers.report results/run-001`.
