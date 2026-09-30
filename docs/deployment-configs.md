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

See the [9B LoRA recipe](../src/spindle/configs/qwen35_9b_lora_16k.py) and [4B FFT recipe](../src/spindle/configs/qwen35_4b_fft_64k.py) for complete examples. Weights are downloaded to `/assets/<model id>`.

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

Validation loads the recipes and checks that they can share one frontend. Backend integration settings are checked when creating a DeploymentConfig at deploy time. Native backend options are checked at engine startup.

The checked-in [deploy_models.sh](../scripts/deploy_models.sh) lists the complete active config set. Add a config path there, then run it. The deployment command owns frontend selection and trainer/inference updates:

~~~bash
./scripts/deploy_models.sh --app my-spindle --env dev
./scripts/deploy_models.sh --refresh-trainer qwen35-9b-lora-16k
./scripts/deploy_models.sh --refresh-inference qwen35-9b-lora-16k
~~~

Frontend, region, environment, secret names, and volume names default in `BaseConfig.platform`. `platform.modal.region` defaults to `None`, which leaves GPU placement unpinned (Modal schedules trainers and inference pools in any region); set it only to restrict placement. Set them in the same Python config using dotted overrides:

```python
from spindle.configs.gpt_oss_20b_lora_64k import Config as Parent


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

All configs deployed together must share platform settings. Use a shared parent config when deploying several models. Explicit `--app`, `--env`, and `--region` flags override the corresponding recipe settings for that invocation. Credentials remain in Modal secrets; recipes contain their names.

One frontend serves all models through Tinker’s `base_model`. When recipes share a model, the first matching recipe in the `spindle deploy` argument list is used. Training also matches the requested LoRA/FFT mode; base sampling uses the first recipe regardless of training mode. To select a specific recipe, pass its name from `/api/v1/spindle/deployments` as `base_model`. The frontend lists the current config set only.

## Update isolation

Trainer apps are named `spindle-trainer-{name}` and inference apps `spindle-inference-{name}`. An inference-only change reuses the trainer app. A trainer-only change reuses the inference app. Source-only upgrades use `--refresh-trainer` / `--refresh-inference`. Updating inference settings also redeploys existing pools for that recipe before publishing the new frontend configuration. An interrupted pool update can be retried with the same command. Changes to region, storage, or secrets refresh both worker apps.

Recipe names must be unique across frontends in the same Modal environment, since worker and pool names include the recipe name. Drain jobs before removing a recipe or making incompatible model, context, or adapter changes.

Trainers receive the frontend App ID at launch and use its shared registry for engine state and reconciliation. A startup failure pauses new starts for five minutes; the failure then expires, allowing subsequent requests and reconciliation to retry. `spindle deployment retry` clears it immediately.

The frontend carries the current configs and exposes them through `deployed_configs`. The CLI reads that function to compare settings, and checks trainer and inference app existence directly with Modal. There is no retained history of earlier configs. Run deploy commands sequentially.

`--app`, `--env`, `--region`, secrets, and storage names are frontend/platform settings. They are not copied onto each recipe.

LoRA pools are named `spindle-lora-{name}`. FFT pools retain the existing `spindle-fft-{session_digest}-latest` or `spindle-fft-{session_digest}-v{version}` names to fit Modal’s 63-character limit. The digest identifies the recipe/client pair; it is not a config revision.

See [validation results](deployment-validation.md) for CPU coverage, live GPU checks, and their limits.
