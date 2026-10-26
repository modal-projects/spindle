# Qwen3.5-9B SWE-Gym supervisor

Deploys the 256k recipe (four B300s training, eight B300s sampling) and keeps the
public SWE-Gym worker up across preemption. The worker itself is
[modal-examples `06_gpu_and_ml/swe_gym`](https://github.com/modal-labs/modal-examples/tree/29c7c4929286d00903339ef863041592a99a2348/06_gpu_and_ml/swe_gym)
at `29c7c4929286d00903339ef863041592a99a2348`. This directory does not carry a
second copy of the sandbox or rollout code.

Before running that script against this recipe, change it in three places:

- `TRAINING["context_tokens"]` to `262144`
- `create_training_client` base model to `swe-gym-9b-256k`
- the reference `create_sampling_client` base model to `swe-gym-9b-256k`

`run/` is local state and is not part of the example. The worker writes
`run/train-job{n}-checkpoint.json` and passes `--resume` after a restart.

On a later attempt the supervisor stops the owned apps, redeploys them, retries
the deployment, and starts the worker again. It also restarts when a client has
no events for 20 minutes or no checkpoint for 3 hours, and it stops for good on
an out-of-memory error, a nonfinite metric, or three failures with no new
checkpoint.

## Deploy, then supervise

From a Spindle checkout, with this repo installed:

```bash
python examples/swe-gym-9b-256k/deploy.py --init-run
export TINKER_API_KEY='your-spindle-api-key'
export SWE_GYM_WORKER=/path/to/modal-examples/06_gpu_and_ml/swe_gym/swe_gym.py
python examples/swe-gym-9b-256k/supervise.py
```

`--init-run` allocates the GPUs, pins the sampler pool at 8 replicas and the
sample proxy at 4–32 containers, and writes `run/launch.json`. A second
`--init-run` refuses to overwrite an existing manifest. The Modal environment
is `MODAL_ENVIRONMENT`.
