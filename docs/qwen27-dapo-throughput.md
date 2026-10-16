# Qwen3.8-27B DAPO throughput comparison

This repeats the eight-client Qwen3.5-9B comparison with `Qwen/Qwen3.8-27B`.
The client loop, 320-row DAPO fixture, group count, sampling seeds, and training
hyperparameters come from `examples/dapo-cost-comparison`.

| Setting | Value |
| --- | --- |
| Adapter | Rank 32, attention and MLP projections |
| Context / output cap | 32,768 / 16,384 tokens |
| Sampling | Thinking enabled, temperature 1, top-p 1, top-k disabled |
| Per-client update | Eight fixed groups of eight responses; no group filtering |
| Training | PPO clipping 0.8/1.28, normalized GRPO advantages, Adam LR 1e-5 |
| Microbatch packing budget | 18,432 tokens/GPU; full 64-response optimizer batch retained |
| Shared trainer | One 8×H200 node, TP1 × DP8, eight adapter slots |
| Inference comparison | Eight TP1 H200 replicas, then eight TP2 H200 replicas |
| Client scheduling | Eight independent loops, starts 150 seconds apart |
| Async overlap | One batch prefetched during training; maximum policy lag one update |
| Planned runs | 30 updates/client on Spindle; one 20-update client on Tinker |

The tokenizer revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Client manifests record dataset and rendered-prompt hashes. Tinker owns its model
revision and implementation. Spindle uses SDK 0.24.1; the actual Tinker run uses
SDK 0.30.4, as in the preceding 9B comparison.

Each self-hosted layout first runs two clients for two updates. Validation requires
that the second rollout overlap the first update for both clients, with exactly
eight groups and no replacements. The pilot deployment is stopped before the full
run, and pilot costs are kept separate. CUDA graphs remain enabled for inference.
The trainer retains the prior optimized benchmark's `TORCH_COMPILE_DISABLE=1`.

Run from this checkout with real Tinker credentials in `TINKER_API_KEY`:

```bash
.venv/bin/python scripts/run_qwen27_dapo.py \
  --output scripts/results/qwen27-dapo/NEW_RUN \
  --spindle-env-file ~/spindle/.env
```

The runner reads only the Spindle key from the env file for self-hosted children.
It starts exactly one Tinker training process. It does not automatically restart
failed paid runs. Use `--resume-existing` with the original output directory to resume self-hosted stages while preserving that Tinker process. Each layout has unique app names, and cleanup stops only those
apps. Per-client errors, an hour without progress, or the experiment deadline stop
the run and preserve its logs. `STOP_NOW` in the output directory requests shutdown.

`status.json` records the active phase and completed client updates. Each client
records rollout start/end, training requests, publication, output tokens, reward,
truncation, and policy lag. Raw group files contain their exact timestamps, prompt
tokens, generated tokens, and behavior logprobs. Trainer logs and GPU container
lifetimes are collected throughout the self-hosted runs.

After all runs finish, `scripts/report_qwen27_dapo.py` writes throughput and cost
tables, PNG/PDF plots, and the results section in `docs/lora_validation.md`.
It reports per-client rollout TPS and aggregate completed-token TPS during the
common active window. GPU cost includes cold starts and staggered idle time.
Tinker costs use observed token counts and published prices, with a range for
unknown input-cache hits; these are estimates rather than invoices.

The trainer activity plots distinguish backend call occupancy from client request
latency. Call occupancy includes synchronization and does not measure GPU hardware
utilization. Preserve the raw logs when comparing queueing with execution.

The first pilot exceeded GPU memory at a 32,768-token packing budget. The rerun
uses 18,432 tokens/GPU with the same optimizer batch and output cap. Tinker
update three encountered a billing pause; the report excludes that update from
latency and throughput, while retaining its tokens in cost accounting.

## Recorded October 8 result

The table below comes from `scripts/results/qwen27-dapo/run-20261008-194808/interim-throughput-cost.json`,
with cutoff `1791496681.4118593`. The original source manifest records base commit
`4f85f09104bde67d483c03f507fac572d3fc7e36` plus local changes. This branch preserves
the experiment checkout, including its runtime changes and diagnostic scripts;
it is not a claim that every later diagnostic edit was active throughout the run.

| Configuration | Aggregate output tokens/s | Cost per million output tokens |
| --- | ---: | ---: |
| Spindle: 8 trainer + 8 inference H200s, 8 clients | 5,502.121175 | $3.666949 |
| Tinker: provider-managed compute, 1 client | 3,636.469541 | $9.768050–$9.791456 |

Spindle's 3,601.67-second window starts when the eighth client begins and excludes
startup and initial staggering. Costs include trainer and inference GPU time,
excluding CPU, memory and storage. The run stopped after a client timeout with
29 total updates (5, 4, 4, 4, 3, 2, 3, 4), rather than the planned 30 per client.
Tinker uses updates 4–9 after a billing pause; its costs are token-price estimates,
with unknown prompt-cache usage, rather than invoice totals. The TP2 stage did
not contribute results. Completed groups crossing the measurement boundary can
include tokens generated before the window or not yet trained.

Use `scripts/report_qwen27_interim.py RUN_DIRECTORY --cutoff 1791496681.4118593`
to recompute the historical table from the local archive. Raw results remain
under the ignored `scripts/results/` directory.
