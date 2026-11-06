# Python deployment configs

A recipe subclasses `BaseConfig` and exports `config = Config()`. Settings are flat, untyped Python attributes. Backend options are ordinary dictionaries.

```python
from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "my-9b"
    model = "Qwen/Qwen3.5-9B-Base"
    max_context_length = 16384
    backend = "miles"
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 16
    trainer_memory_mib = 65536
    trainer_max_clients_per_instance = 6
    inference_gpu = "H200"
    inference_max_replicas = 8
    miles_cfg = {
        "model_type": "qwen3.5-9B",
        "tensor_model_parallel_size": 4,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head",
        ],
    }
    sglang_cfg = {"max_running_requests": 32}


config = Config()
```

See the [9B LoRA recipe](../src/spindle/configs/qwen35_9b_lora_16k.py) and [4B FFT recipe](../src/spindle/configs/qwen35_4b_fft_64k.py) for complete examples. `spindle deploy` downloads weights to `/assets/<model id>` in the background; requests that arrive before the download finishes wait for it.

## Variants

Override ordinary attributes directly. Use dotted `overrides` to change individual backend options:

```python
from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "my-9b-more-memory"
    trainer_memory_mib = 98304
    overrides = {"miles_cfg.max_tokens_per_gpu": 8192}


config = Config()
```

Each parent's settings and overrides apply before its child's. Constructor fields and overrides apply last: `Config(trainer_gpu="H200", overrides={"sglang_cfg.max_running_requests": 16})`. Assigning a dictionary or list replaces that value; `trainer_env = {}` clears inherited environment settings. Instances own independent copies of mutable values and can also be edited directly.

`trainer_gpu` and `inference_gpu` take one GPU type or a list in preference order, such as `["H200", "B300"]`. Modal schedules the first type with capacity. GPU counts and parallelism settings apply to every entry, so list only types the recipe fits on.

When several clients share a trainer, it trains all of their queued batches together by default, and each client waits for the whole step to finish. For RL, where clients sample at different times, set `trainer_max_clients_per_batch = 1`. The trainer then runs one client at a time, so each client can go back to sampling as soon as its own step is done.

## Backend options

| Setting | Consumed by |
| --- | --- |
| `trainer_*`, `inference_*` | Modal GPU/CPU/memory allocation, scaling, timeouts and Spindle admission limits |
| `megatron_cfg` | Existing Megatron `EngineModelConfig`; provider, optimizer and distributed options use its native dictionaries |
| `miles_cfg` | Existing `MilesBackendConfig`; `cli_options` supplies additional Miles arguments |
| `sglang_cfg` | SGLang `ServerArgs` |

Modal and backend libraries validate their own options. Spindle checks integration requirements such as trainer slot capacity, supported training modes, and parallelism agreeing with allocated GPUs. It supplies managed model paths, context length and adapter settings; conflicting backend overrides are rejected.

`BaseConfig` does not enforce field types or reject arbitrary attributes. Extra backend options belong in the corresponding backend dictionary. A misspelled top-level attribute is ordinary Python data and may be unused.

Miles argument conversion lives in [miles_arguments.py](../src/spindle/backends/miles_arguments.py). SGLang receives `ServerArgs(**settings)` in its [worker entrypoint](../src/spindle/inference/sglang.py). Backend libraries validate native options when workers start; frontend config imports remain CPU-only.

### Miles options

Besides the LoRA settings, `miles_cfg` accepts `megatron_to_hf_mode` (`bridge`, the default, or `raw`: the converter Miles uses to export weights, for models without a Megatron-Bridge mapping) and `dynamic_batch_size`.

### Images

`trainer_image` selects a named trainer image instead of the backend default; the names are listed in [deployment_apps.py](../src/spindle/providers/modal/deployment_apps.py). The multi-LoRA actor runs on Miles v0.1.1 and v0.1.2.

## Prepare and launch

~~~text
load(config.py) → recipe: BaseConfig
  → DeploymentConfig.create(recipe)
  → deploy named trainer and inference apps if they are missing or changed
  → deploy the frontend with the current config set
~~~

The launcher consumes the saved settings. It does not reparse backend configuration or add another set of backend defaults. JSON decoding in a trainer reconstructs the saved config, without importing the author's config file.

| File | Responsibility |
| --- | --- |
| [configuration.py](../src/spindle/configuration.py) | BaseConfig defaults, inheritance and overrides |
| [deployments.py](../src/spindle/deployments.py) | Python recipe loader and DeploymentConfig |
| [backends/deployment.py](../src/spindle/backends/deployment.py) | Backend settings used when creating a DeploymentConfig |
| [megatron_runtime/common/settings.py](../src/spindle/backends/megatron_runtime/common/settings.py) | Shared Megatron ownership rules and constructor dictionaries |
| [deployment_apps.py](../src/spindle/providers/modal/deployment_apps.py) | Trainer, inference, and pool app builders |
| [deployment_configs.py](../src/spindle/providers/modal/deployment_configs.py) | Read saved configs and frontend platform settings |
| [deployment_cli.py](../src/spindle/deployment_cli.py) | Deploy trainer and inference apps by calling the builders, then deploy the frontend |

## Multi-node Miles

[qwen38_27b_lora_256k.py](../src/spindle/configs/qwen38_27b_lora_256k.py) configures two nodes with eight H200s per node, TP2 and CP8. The trainer app uses Modal's clustered launcher and RDMA. Every rank mounts the same volumes; rank 0 starts the engine, and the other ranks join Ray using the launcher merged in #39. The driver receives the Ray address. GPU counts cannot be independently overridden through Miles options.

This path has CPU construction/topology tests. This refactor has not been redeployed or tested on multiple GPU nodes.

## Deploy and update

~~~bash
spindle config init --preset qwen35-9b-lora-16k > my_model.py
spindle config validate my_model.py
spindle deploy my_model.py
~~~

Bundled configs deploy by name, e.g. `spindle deploy qwen35-9b-lora-16k gpt-oss-20b-lora-32k`, and `spindle deploy --all` deploys every bundled config. With no arguments in a terminal, `spindle deploy` asks which models to deploy; without a terminal it exits instead of deploying.

Validation loads the recipes and checks that they can share one frontend. Backend integration settings are checked when creating a DeploymentConfig at deploy time. Native backend options are checked at engine startup.

The checked-in [deploy_models.sh](../scripts/deploy_models.sh) lists the complete active config set. Add a config path there, then run it. The deployment command owns frontend selection and trainer/inference updates:

~~~bash
./scripts/deploy_models.sh --app my-spindle --env dev
./scripts/deploy_models.sh --refresh-trainer qwen35-9b-lora-16k
./scripts/deploy_models.sh --refresh-inference qwen35-9b-lora-16k
~~~

Frontend, region, environment, secret names, and volume names default in `BaseConfig.platform`. `platform.modal.region` defaults to `None`, which leaves GPU placement unpinned (Modal schedules trainers and inference pools in any region); set it only to restrict placement. The frontend server, which runs on CPU, is placed in that region, or in `us-east` when it is unset; this never affects where GPUs are scheduled. Set them in the same Python config using dotted overrides:

```python
from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent


class Config(Parent):
    overrides = {
        "platform.frontend": "my-spindle",
        "platform.modal.environment": "dev",
        "platform.modal.region": "us-east",
        "platform.secrets.api": "my-api-secret",
        "platform.storage.checkpoints": "my-checkpoints",
    }


config = Config()
```

The frontend's URL is `https://<workspace>-<environment>--<frontend>-server.modal.run`. To choose the part after `--`, set `"platform.url_label"`; for example `"prod"` gives `https://<workspace>-<environment>--prod.modal.run`. Changing the label changes the URL clients use.

Modal proxy auth is enabled by default: Modal rejects any request without a [proxy token](https://modal.com/docs/guide/webhook-proxy-auth) allowed for the environment, before it reaches Spindle. Set `"platform.requires_proxy_auth": False` to opt out. Clients send the token as headers next to their Tinker API key:

```python
service = tinker.ServiceClient(
    default_headers_overrides={"Modal-Key": token_id, "Modal-Secret": token_secret}
)
```

Tinker SDKs older than 0.36 call this argument `default_headers`. Scripts that create their own `ServiceClient`, such as tinker-cookbook's, can run through [`with_proxy_auth.py`](../examples/shared-server/with_proxy_auth.py), which adds the headers to every client the script creates.

All configs deployed together must share platform settings. Use a shared parent config when deploying several models. Explicit `--app`, `--env`, and `--region` flags override the corresponding recipe settings for that invocation. Credentials remain in Modal secrets; recipes contain their names.

One frontend serves all models through Tinker’s `base_model`. If several recipes serve the same model, requests by model name go to the one with the longest context. To use a specific recipe, pass its name as `base_model`. `get_server_capabilities` lists every recipe and its context length.

`GET /api/v1/spindle/queues` describes each recipe's training queue. Each trainer holds up to `trainer_max_clients_per_instance` training clients, and a recipe runs at most `trainer_max_instances` trainers. When every slot is in use, a new client waits in line instead of failing. The client that has waited longest gets the next free slot. It keeps waiting until a slot frees, with no time limit.

```json
{"name": "qwen35_2b_lora_32k", "base_model": "Qwen/Qwen3.5-2B", "parameterization": "lora",
 "backlog": {"count": 1, "oldest_age_s": 42.0},
 "slots": {"used": 6, "free": 0, "per_trainer": 6},
 "trainers": {"running": 1, "starting": 1, "max": 4},
 "recent": {"window_s": 3600.0, "added": 4, "placed": 3, "released": 2,
            "wait_s": {"p50": 12.0, "p95": 80.0, "max": 95.0}}}
```

- `backlog` counts training clients waiting for a slot, and how long the oldest has waited.
- `slots` counts clients on running trainers and the open slots left on them.
- `trainers` counts trainers that are running, trainers that are still booting, and the most the recipe allows.
- `recent` covers the last `window_s` seconds. `added` counts clients that asked for a slot, `placed` counts clients that got one, and `released` counts slots that opened up. A slot opens when a client unloads its model or its session closes or goes idle. Models lost when a trainer stops are not counted, since no slot opens for anyone else. `wait_s` gives the median, 95th percentile and longest wait of the clients placed in the window, or `null` if none were placed.

A client can start right away when `slots.free` is greater than `backlog.count`. Otherwise it waits for a trainer to boot, or for another client to finish or go idle for `session_idle_timeout_s`. While it waits, the Tinker SDK shows its place in line, such as `all trainer slots for Qwen/Qwen3.5-2B are in use; waiting for one to free up (2nd in line, 1m 10s so far)`. A client that stops waiting, for example because its script exited, loses its place within 30 seconds. The queue response and each frontend's view of the line can be a few seconds old, so a new client can occasionally take a slot just as it frees up.

When no slot is free and no trainer is booting, `recent.released` shows how fast slots open up. For example, 4 slots released in the last hour means a slot opens about every 15 minutes. `backlog.count` shows how many other clients are waiting for those slots.

## The frontend registry

Every recipe a frontend serves is one `<app_id>/<recipe>.json` config file in the shared `spindle-deployments` Modal Volume, under a folder named after the frontend's app ID. `spindle deploy` deploys the trainer and inference apps that are missing or changed, then writes the listed configs to the registry and publishes the frontend with only its platform and lifecycle settings. The frontend reads the registry, so recipes not listed in a deploy keep serving. A deploy that changes platform settings must list every recipe the frontend serves, since unlisted ones would keep apps built for the old settings.

Upgrading a frontend that still ships a configs file needs no extra step: `spindle deploy` on the new version writes every recipe the old frontend serves to the registry before the swap, so nothing drops out and unchanged apps are not deployed again.

## Add or remove a config on a running frontend

Anyone with the frontend's API key can add and remove configs. There are two ways to add configurations: SDK or REST API.

### SDK

~~~bash
uv run spindle add qwen35-9b-lora-16k-h200 --base qwen35-9b-lora-16k \
  --trainer_gpu=H200 --miles_cfg.max_tokens_per_gpu=8192
~~~

`spindle add` waits until the config's apps are ready. To remove a config:

~~~bash
uv run spindle remove qwen35-9b-lora-16k-h200
~~~

If models still train on it, `spindle remove` lists them and prompts the user first; `--force` skips the interactive prompt.

### REST API

`POST /api/v1/spindle/deployments` writes a config into a running frontend's registry and deploys its trainer and inference apps in the background, without redeploying the frontend. Send a new `name` with either the deployed config to start from as `base` plus the settings to change as `overrides` (the same dotted keys as a config's `overrides`), or a full `recipe`:

~~~bash
curl -sS -i -X POST "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "qwen35-9b-lora-16k-h200",
       "base": "qwen35-9b-lora-16k",
       "overrides": {"trainer_gpu": "H200", "miles_cfg.max_tokens_per_gpu": 8192}}'
~~~

`DELETE /api/v1/spindle/deployments/{name}` removes a config and stops its trainer and inference apps, and `GET /api/v1/spindle/deployments` lists them. If models still train on the config, `DELETE` returns HTTP 409 listing them; add `?force=true` to remove it anyway, which ends their training.

Configs deployed before this API are already registered and listed. To move a config kept in a Python file onto the API, send its fields as `recipe` (`vars(config)`).

Each listed config reports `apps` as `"deploying"`, `"ready"`, `"failed"` (with `apps_error`) or `"missing"`. Sending the same config again retries a failed deploy.

If you find a config that clearly beats the current baselines, please upstream it to Spindle as a bundled config.

## Update isolation

Trainer apps are named `{frontend}-trainer-{name}` and inference apps `{frontend}-inference-{name}`, where `{frontend}` is the frontend app name (`spindle` by default, or the `--app` value). An inference-only change reuses the trainer app. A trainer-only change reuses the inference app. Source-only upgrades use `--refresh-trainer` / `--refresh-inference`. Updating inference settings also redeploys existing pools for that recipe before publishing the new frontend configuration. An interrupted pool update can be retried with the same command. Changes to region, storage, or secrets refresh both worker apps.

Every app a deployment creates starts with its frontend name, so two deployments with different `--app` values can share one Modal environment and the same recipe names. Drain jobs before removing a recipe or making incompatible model, context, or adapter changes.

Trainers receive the frontend App ID at launch and use its shared registry for engine state and reconciliation. A startup failure pauses new starts for five minutes; the failure then expires, allowing subsequent requests and reconciliation to retry. `spindle deployment retry` clears it immediately.

The CLI reads the current configs straight from the frontend's registry folder to compare settings, and checks trainer and inference app existence directly with Modal. There is no retained history of earlier configs. Run deploy commands sequentially.

`--app`, `--env`, `--region`, secrets, and storage names are frontend/platform settings. They are not copied onto each recipe.

LoRA pools are named `{frontend}-lora-{name}`. FFT pools are named `{frontend}-fft-{session_digest}-latest` or `{frontend}-fft-{session_digest}-v{version}` to fit Modal’s 63-character limit. The digest identifies the recipe/client pair; it is not a config revision.

`spindle stop --app my-spindle --env dev` stops the frontend and every app whose name starts with `my-spindle-trainer-`, `my-spindle-inference-`, `my-spindle-lora-`, or `my-spindle-fft-`, ending any runs on them. Deployments made with `--app` on Spindle 0.1.5 or older named their worker and pool apps `spindle-...`; stop those with `modal app stop` after upgrading. It also deletes the stopped frontend's folder from the registry Volume, leaving other frontends' folders alone: the next `spindle deploy` under that name creates a new frontend app with an empty registry, so it serves only the configs it lists.

See [validation results](deployment-validation.md) for CPU coverage, live GPU checks, and their limits.
