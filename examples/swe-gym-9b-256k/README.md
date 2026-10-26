# Qwen3.5-9B SWE-Gym, 256k, eight clients

Supervisor and worker for the 63-hour SWE-Gym run: four B300s training and
eight B300s sampling, eight independent LoRA clients, 100 batches each. The
GPU recipe is `qwen35_9b_instruct_lora_256k`. This launcher gives it a
dedicated frontend and a one-day idle timeout.

Spindle saves and restores a client only when this supervisor asks. On a later
attempt the supervisor stops the owned apps, redeploys them, clears the
deployment failure, and starts the worker again. The worker writes
`run/RESTART_REQUIRED.json` on `model_lost` or a timeout. The supervisor also
restarts when a client has no events for 20 minutes or no optimizer or
checkpoint progress for 3 hours. It stops for good on an out-of-memory error,
a nonfinite metric, or three failures that produce no new checkpoint.

Resume is per client. A finished batch writes
`run/eight-job{n}-checkpoint.json` with the `save_state` path. The next worker
opens that client's trainer from it. Client `n` starts its rollout 450 seconds
after client `n-1` logs `ready`. There is no step barrier between clients.

The October 9 run used the `training-team` Modal environment and the frontend
`spindle-swe-256k-tp1-20261009`. This example uses `MODAL_ENVIRONMENT` and
`spindle-swe-gym-9b-256k`, so it does not replace that deployment or the shared
`spindle` app. `run/` is local state and is not part of the example.

## Deploy, then supervise

From a Spindle checkout, with this repo installed:

```bash
python examples/swe-gym-9b-256k/deploy.py --init-run
export TINKER_API_KEY='your-spindle-api-key'
python examples/swe-gym-9b-256k/supervise.py
```

`--init-run` allocates 4 trainer B300s and 8 sampler B300s, pins the sampler
pool at 8 replicas and the sample proxy at 4–32 containers, and writes
`run/launch.json`. It does not return until the apps are deployed. A second
`--init-run` refuses to overwrite an existing manifest.

The worker also needs `tinker`, `transformers`, `numpy`, `orjson`, `datasets`,
and the [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) harness (`swegym`) on
`PYTHONPATH`. `psutil` is optional; without it the supervisor skips host
memory and CPU lines.

`data/validated_workload_ids.json` lists the 115 tasks this run trained on. The
worker loads those rows from the public
[SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) dataset, in
that order. Grading runs in a fresh Modal sandbox
(`swe-gym-9b-256k-sandboxes`, 16 GiB, network blocked).

The tokenizer is `Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
