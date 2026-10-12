# Benchmark 27B: Tinker vs Spindle

The DAPO workload behind the October 8, 2026 Qwen3.8-27B comparison, packaged
for new experiments. Edit `deployment.py` for infrastructure and `config.json`
for the shared workload. Both providers run the same `client.py`.

| Setting | Value |
| --- | --- |
| Model / LoRA | `Qwen/Qwen3.8-27B`, rank 32, attention + MLP; no unembedding |
| Context / output cap | 32,768 / 16,384 tokens; thinking enabled |
| Per-client update | 8 prompt groups × 8 responses; no dynamic filtering |
| Sampling | Temperature 1, top-p 1, top-k disabled; seed 4242 + group index |
| Training | Normalized group advantages, token-mean PPO, clip 0.8/1.28, Adam LR 1e-5 |
| Trainer | 8 H200s, TP1/DP8, 8 adapters, full activation recomputation |
| Packing budget | 18,432 tokens/GPU; the full 64-response optimizer batch is retained |
| Inference | 8 replicas × 1 H200, TP1, CUDA graphs enabled |
| Scheduling | 8 independent clients, 150-second stagger; 1 prefetched batch, policy lag ≤1 |
| Planned run length | 30 updates/client on Spindle, 20 updates on one Tinker client |

`dapo.jsonl` is the fixed 320-row **input dataset**, not generated results. It
retains row indices from [open-r1/DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed).
The client pins the tokenizer revision and records dataset/config/rendered-prompt
hashes. It grades the final boxed numeric answer. Equal-reward groups remain in
the training batch. Adapter initialization and provider model revisions are not
guaranteed to match; compare reward, response lengths and truncation too.

## Install

Run from the repository root with Python 3.12 and `uv`. Use separate environments:
the historical Spindle client used Tinker SDK 0.24.1, while Tinker used 0.30.4.
Keep those versions visible when comparing results.

```bash
uv venv --python 3.12 .venv-bench-spindle
uv pip install --python .venv-bench-spindle/bin/python -e . \
  -r examples/benchmark-27b-tinker-vs-spindle/requirements-spindle.txt
uv venv --python 3.12 .venv-bench-tinker
uv pip install --python .venv-bench-tinker/bin/python \
  -r examples/benchmark-27b-tinker-vs-spindle/requirements-tinker.txt
```

## Spindle

Complete the repository's [Modal credentials and secrets setup](../../README.md#2-configure-modal-and-secrets-once).
The recipe uses the standard `spindle-api`, `spindle-proxy` and Hugging Face
secrets, with dedicated benchmark app/volume names. Set the environment, deploy,
and copy the frontend URL printed by deployment:

```bash
export MODAL_ENVIRONMENT=your-environment
.venv-bench-spindle/bin/python -m spindle.deployment_cli deploy \
  examples/benchmark-27b-tinker-vs-spindle/deployment.py \
  --env "$MODAL_ENVIRONMENT"
export TINKER_BASE_URL='YOUR_DEPLOYED_FRONTEND_URL'
export TINKER_API_KEY='YOUR_SPINDLE_KEY'
```

Start with a two-client, two-update pilot using the same workload:

```bash
.venv-bench-spindle/bin/python examples/benchmark-27b-tinker-vs-spindle/run_spindle.py \
  --root examples/benchmark-27b-tinker-vs-spindle/runs/pilot \
  --base-url "$TINKER_BASE_URL" --environment "$MODAL_ENVIRONMENT" \
  --clients 2 --updates 2 --stagger-seconds 150 \
  --owned-apps spindle-benchmark-27b spindle-trainer-benchmark-27b \
    spindle-inference-benchmark-27b spindle-lora-benchmark-27b
```

Inspect `client-*/events.jsonl`: each client's second rollout should overlap its
first training request, and clients should advance without a cross-client barrier.
The supervisor stops the named apps on completion or failure, so **deploy again**
before the full run. Replace `runs/pilot` with a fresh `runs/spindle`, and use
`--clients 8 --updates 30`; keep the 150-second stagger. The stagger clock starts
after client zero's sampling warmup, matching the original experiment.

Do not share these app names between simultaneous experiments. When changing the
recipe name or frontend, update `--owned-apps` too. The supervisor stops only
that explicit list. Ctrl-C, `STOP_NOW` in the run directory, client failure, an
hour without client events, or the default 12-hour deadline triggers cleanup.
`status.json` records progress, resource-collection errors and cleanup results.

## Tinker

In a **separate terminal**, use the real Tinker key. Passing an empty base URL
explicitly selects the SDK default endpoint:

```bash
unset TINKER_BASE_URL
export TINKER_API_KEY='YOUR_TINKER_KEY'
.venv-bench-tinker/bin/python examples/benchmark-27b-tinker-vs-spindle/client.py \
  --base-url '' --updates 20 \
  --output examples/benchmark-27b-tinker-vs-spindle/runs/tinker
```

There is no automatic paid retry or checkpoint resume. Use a fresh directory for
each run. Client outputs contain manifests, timing/token events and compressed
generated groups; `runs/` is ignored by Git.

## Reporting

The reporter needs only the Python standard library and prints JSON, without
plots or edits to repository docs:

```bash
python examples/benchmark-27b-tinker-vs-spindle/report.py \
  --spindle examples/benchmark-27b-tinker-vs-spindle/runs/spindle \
  --tinker examples/benchmark-27b-tinker-vs-spindle/runs/tinker/events.jsonl
```

For completed Spindle runs, the common active window starts at the last client's
first rollout and ends when the first client finishes. For a live/incomplete
run, pass `--cutoff UNIX_TIMESTAMP` explicitly, after all clients have started.
The report counts groups finishing within that window and clips recorded GPU
lifetimes to it. This excludes startup, initial staggering, CPU, memory and
storage cost. Boundary-crossing groups can include tokens generated before the
window or not yet trained. Short-lived containers may escape resource polling;
inspect collection errors and lifetimes before making a cost claim.

Tinker uses consecutive completed updates, including their training/publication
time, with optional `--tinker-first-update` / `--tinker-last-update`. It does not
silently skip billing pauses. Prices at the top of `report.py` are the historical
October 8 rates; update them for new runs. Tinker costs show the full input-cache
uncertainty range and are estimates, not billing ground truth.

## Historical table and limits

The quoted **5,502 TPS / $3.67** Spindle and **3,636 TPS / $9.77** Tinker table
came from an **interim, incomplete run**, `run-20261008-194808`. Spindle's roughly
one-hour window ended at `1791496681.4118593`, with 2–5 completed updates per
client and about 688 TPS/client. Tinker used updates 4–9 after a billing pause;
$9.77 was the lower end of a $9.768–$9.791 estimate.

The small [Tinker reference extract](reference/README.md) retains the measured
events needed to recompute its row without a new paid run:

```bash
python examples/benchmark-27b-tinker-vs-spindle/report.py \
  --tinker examples/benchmark-27b-tinker-vs-spindle/reference/tinker-events.jsonl \
  --tinker-first-update 4 --tinker-last-update 9
```

Raw Spindle results and plots are intentionally omitted. With the original run
directory, supply its `tp1-full/training` as `--spindle`, the cutoff above, and
the reference Tinker arguments to recompute both rows.

This example preserves the original client loop, workload and GPU layout, and
runs against the checked-out Spindle backend. The historical checkout was based
on `4f85f09104bde67d483c03f507fac572d3fc7e36` with uncommitted runtime changes:
request/response transport optimizations, optimizer/publication priority,
4-CPU/8-GiB frontend and sampling workers, and forwarding `TORCH_COMPILE_DISABLE`
to Ray workers. Those backend changes are outside this example. The recipe
retains the compile-disable request, but main at the time of this PR does not
explicitly forward it to existing Ray workers. New throughput measurements are
required; these scripts do not guarantee the historical TPS on current main.

For TP2 inference experiments, change `inference_gpus_per_node` and
`sglang_cfg.tp_size` together, and update admission limits as appropriate. The
original TP2 stage never completed and contributed no numbers to the table.

Validate the deployment recipe locally before deploying:

```bash
.venv-bench-spindle/bin/python -m spindle.deployment_cli config validate \
  examples/benchmark-27b-tinker-vs-spindle/deployment.py
```
