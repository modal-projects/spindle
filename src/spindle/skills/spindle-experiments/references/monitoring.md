# Monitoring and evaluation

## Client-side

`forward_backward`/`optim_step` futures return metrics: `loss:sum`, `loss:mean`, `tokens:sum`, `n_sequences:sum`, `response_length:mean`. `scripts/wandb_rl_example.py` logs reward, response length, `time/sample_s`, `time/train_s`, `time/step_s` to W&B (purely client-side).

## OTLP

Set `OTEL_*` on the `spindle-api` secret (or `telemetry_secret` on `spindle.run`); search `service:spindle` in Datadog APM.

- `spindle.command.forward_backward`: command lifetime; waiting shows as gaps, no queue span.
- `spindle.trainer.forward_backward`/`optim_step`: physical batches; count these to avoid double-counting shared ops.
- `spindle.sample`/`sample.attempt`: `version_requested`, `version_served_*`, `sglang.queue_s`, `prefill_s`, `post_prefill_to_finish_s`.
- `spindle.trainer.state` gauge: current op per lane every 5 s; operation state, not GPU utilization.
- Labels: `user_metadata` `run_id`/`attempt_id` → `spindle.run_id`/`run_attempt_id`. Same `run_id` across recovery, fresh `attempt_id` per replacement. A batch is labeled only when all commands share it; the gauge and base sampling are unlabeled.

## Startup marks

Each container logs `{"event": "spindle_startup_mark", ...}` per phase:

```bash
uv run modal app logs spindle-trainer-<config> \
  | grep -o '{"event": "spindle_startup_mark".*}' \
  | jq -r '[.component, .mark, .since_start_s, .phase_s] | @tsv'
```

## Modal side

`modal app list`/`modal app logs` + GPU metrics; apps tagged `_modal_source=spindle`, `spindle_role`. Leftover `spindle-fft-...` samplers bill while warm; frontend stop does not stop them.

## Cadence

Check between steps. A step materially longer than the run's prior timings is the trigger to investigate, starting with `spindle.trainer.state` for what the trainer was doing in the gap.

## Evaluation

- Eval the base model first (codegolf step 0 was eval only); deltas need it.
- Eval on a pinned publication; the latest pool can serve newer weights mid-eval.
- pass@k/best-of-k: codegolf uses 16 stochastic samples per problem (one answer = 6.25 pts); keep the count fixed across steps.
- Track response length, truncation, diversity, entropy alongside aggregates; eval sets in the repo's runs are small (16 problems).
