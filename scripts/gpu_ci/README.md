# DAPO GPU validation

This manual GitHub Actions workflow deploys the checked-out Spindle commit on
Modal and runs real Qwen3.5-9B multi-LoRA training. It allocates **24 H200 GPUs**:
one eight-GPU trainer (TP1/DP8) and eight two-GPU inference replicas (TP2).
It is opt-in, not an automatic GPU job on every PR.

| Mode | Clients | Updates/client | Start policy | Deadline |
| --- | ---: | ---: | --- | --- |
| `correctness` | 8 | 3 | Warm client zero, create the others, then wait for all eight before rollouts | 1 hour |
| `performance` | 8 | 30 | Launch every 150s after client zero is ready | 3 hours |

The correctness barrier intentionally exercises shared contention. The short
run cannot establish full-run throughput. Performance mode fails if end-to-end
output throughput falls **more than 15% below a reviewed reference**. The gate
uses total generated training-rollout tokens divided by wall time from the first
client's training-loop start to the last client's training-loop finish. It includes
sampling, training, publication, staggered ramp-up and drain, but excludes initial
warmup and final probes. The same window is used for the reference and test run.
Tokens/sec remains sensitive to generated lengths; inspect the reported lengths,
truncation, and rewards when diagnosing failures.

The old benchmark numbers are context, not an automatically accepted baseline:
the new harness needs its own full reference run. Correctness mode has no
performance threshold.

## One-time setup

1. Create a dedicated Modal environment named `spindle-ci`. In that environment,
   configure the standard `spindle-api` secret (`TINKER_API_KEY`, starting with `tml-` as required by the SDK),
   `spindle-proxy` secret (`MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET`),
   and `huggingface-secret` (`HF_TOKEN`) using the [server setup](../../README.md#shared-deployment-quick-start).
   The runner uses the standard model-assets, checkpoint, and bulletin volumes
   in this environment, creating them if absent. Initial model download/image
   builds may need a separate warmup run; cold starts count against the deadline.
2. Create a GitHub environment named `gpu-ci` with secrets `MODAL_TOKEN_ID`,
   `MODAL_TOKEN_SECRET`, and `SPINDLE_API_KEY`. The API key must match the Modal
   `spindle-api` secret. Set `HF_TOKEN` as well if required for tokenizer access.
   Give the Modal token permission to deploy, inspect tasks/placement dictionaries,
   and stop apps in the CI environment. Restrict workflow execution to trusted
   branches/maintainers using GitHub environment protection rules.
3. Merge the workflow into the default branch so GitHub exposes its manual trigger.
   Select the branch to test in Actions, or run:

   ```bash
   gh workflow run gpu-dapo.yml --repo modal-projects/spindle \
     --ref YOUR_BRANCH -f mode=correctness
   ```

Only one GPU validation runs at a time. No Tinker billing key is needed: the
standard Tinker SDK client connects to this run's Spindle endpoint.

## Establishing the reference

1. Run a known-good commit with `mode=performance` and `record_reference=true`:

   ```bash
   gh workflow run gpu-dapo.yml --repo modal-projects/spindle \
     --ref KNOWN_GOOD_BRANCH -f mode=performance -f record_reference=true
   ```

   Locally, add `--mode performance --record-reference` to `prepare`.
   This still runs every correctness check. Only a successful run and cleanup
   produce `reference.json` in the results artifact.
2. Inspect the run's logs, topology, reward/length statistics, and timing plots.
   Prefer repeated runs to check noise before choosing a representative reference.
3. Commit the reviewed artifact as `scripts/gpu_ci/reference.json`. It records
   measured throughput, the source commit/run, workload, deployment settings,
   tokenizer/prompt identity, and client package versions. Baseline updates are
   explicit reviewed changes; CI never overwrites this file.
4. Run subsequent performance checks with `record_reference=false` (the default).
   A missing or incompatible workload/topology reference fails **before GPU
   deployment**; prompt/package incompatibility fails after collection. The floor
   is `reference_throughput * 0.85`; exactly 85% passes. Below it, the job fails and
   retains the comparison in `summary.json` and `summary.md`.

There is intentionally no fabricated reference checked in yet. Changing workload,
packages or topology requires a new reference. The base-weight cache is not
independently fingerprinted; keep it consistent when establishing/comparing runs.

## Local invocation

Use Python 3.12 and the repository's locked environment, then install
`scripts/gpu_ci/requirements.txt`. Export the Modal credentials and the matching
`TINKER_API_KEY`. These commands allocate GPUs; `prepare` alone does not:

```bash
uv sync --locked --python 3.12
uv pip install --python .venv/bin/python -r scripts/gpu_ci/requirements.txt
export PYTHONPATH=src:.
export MODAL_ENVIRONMENT=spindle-ci
.venv/bin/python -m scripts.gpu_ci.run prepare \
  --output scripts/results/my-gpu-ci --mode correctness
.venv/bin/python -m scripts.gpu_ci.run run --output scripts/results/my-gpu-ci
```

## What passes and fails

The gate requires all eight distinct models to finish every expected update,
use the same workload/prompts, share one trainer instance, and have the expected
live H200 allocation. It checks finite losses/optimizer metrics, successful
optimizer updates, generated/trained token counts, token/logprob alignment,
one-update maximum policy lag, and unique sampler publications owned by each
model. Every client must see at least one mixed-reward group and nonzero gradient;
a wholly zero-signal run fails as inconclusive.

A fixed-token logprob probe before/after training checks that the published
policy output changed. This is a functional check, not a weight checksum or a
proof of numerical isolation: inference nondeterminism can also change logprobs.
Exact adapter-isolation/numerical parity remain separate tests. Rewards do not
have to increase in three updates. The final probe publishes/samples the final
weights outside the measured training loop; it adds small real resource usage.

`config.json` retains the prior fixed-group DAPO workload: rank 32, thinking,
32k context, 16k output cap, 8 groups × 8 responses, PPO clipping 0.8/1.28,
learning rate 1e-5, and one prefetched batch. Only the update count changes by mode.
`dapo.jsonl` is the same 320-row numeric-answer fixture used in the prior runs,
from [open-r1/DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed).
The tokenizer revision, client packages, prompt IDs, fixture hash, and checkout
SHA are recorded. Base weights follow Spindle's shared asset-volume cache;
their revision is not independently verified by this gate. Changing the cache
requires recalibrating any performance/reward baseline.

## Results and cleanup

Actions uploads manifests, client event logs, deployment/backend logs, placement
and GPU topology snapshots, `summary.json`, a Markdown summary, and reward/time
curves. Raw rollout files remain local and are excluded from the artifact to
avoid uploading hundreds of megabytes. `diagnostics.json` reports backend call
occupancy and engine queue time when timing logs are present; missing/partial
logs are explicitly marked and never treated as zero queueing. Sampling duration
includes client processing and inference waiting; it is not pure decode time.
`resources.json` reports GPU-seconds for observed tasks once their final lifetimes
are available, or null if unavailable. It is not a complete billing invoice.

Every run has unique app names with its expiry timestamp. Cleanup runs in the
runner's `finally` block and an Actions `always()` step. Before GPUs are deployed,
the runner deploys and executes a CPU-only `spindle-gpu-ci-janitor` app. Its
10-minute schedule stops only expired apps matching the exact GPU CI name format
in the dedicated environment. It survives loss of the GitHub runner, and is
intentionally kept deployed. This backstop can leave resources running until the
deadline plus a scheduling interval; it is not an instantaneous spend cap.
Volumes and app-scoped metadata are retained for inspection, not automatically deleted.

CPU tests exercise validation failures, workload preservation, topology checks,
and cleanup ownership without provisioning resources:

```bash
uv run pytest -q tests/gpu
```
