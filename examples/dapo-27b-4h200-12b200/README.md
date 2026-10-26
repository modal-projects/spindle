# Qwen3.8-27B DAPO: 4×H200 train, 12×B200 sample

Launcher and client for the DAPO-math row in the Training API post. One
4-GPU H200 trainer (TP1, DP4) and three 4-GPU B200 sampling replicas (TP4).
Each client runs its own rollout, train, and publish loop. The second client
starts 150 seconds after the first logs `ready`. There is no step barrier
between clients.

`config.json` is the 30-update client the launcher loads by default. The
1/2/4/8 client utilization sweep used the same settings with `"updates": 5`,
deployed the same GPUs with `--lora-slots 16`, and started rollouts from one
shared gate (`launch_clients.py --mode gate --stagger-seconds 10`).

Secrets and the Modal environment come from the normal Spindle setup
(`spindle-api`, `spindle-proxy`, and `MODAL_ENVIRONMENT`). The original run
used a private environment; those secret names are not in this example.

## Deploy

From a Spindle checkout, with this repo installed:

```bash
python examples/dapo-27b-4h200-12b200/deploy.py --launch
export TINKER_BASE_URL="$(python -c 'import json; print(json.load(open("examples/dapo-27b-4h200-12b200/endpoint.json"))["base_url"])')"
export TINKER_API_KEY='your-spindle-api-key'
```

`--launch` allocates 4 H200s and 12 B200s and does not return until the apps
are deployed.

## Two clients

```bash
python examples/dapo-27b-4h200-12b200/launch_clients.py \
  --clients 2 --stagger-seconds 150 --mode ready
```

Client dependencies, in addition to Spindle: `tinker`, `transformers`, `numpy`.

Training is rank-32 LoRA, 8 groups of 8 samples, 16,384 generation tokens,
32,768 context. `miles_cfg.max_tokens_per_gpu` is 18,432 because packing the
full 32,768 context in one microbatch OOMed. Sampling admission is 80 running
requests and 8 queued requests per replica.
