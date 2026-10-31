# Deploy

## Connect

`tinker.ServiceClient(base_url, api_key)` or env `TINKER_BASE_URL`/`TINKER_API_KEY`. Clients need only the URL and API key.

## Install

`uv add 'modal-spindle @ git+https://github.com/modal-projects/spindle.git'`. Python 3.12 for scoped runs and deploys. Select the Modal workspace and `MODAL_ENVIRONMENT` before creating secrets.

## Credentials

| Credential | Purpose |
| --- | --- |
| Modal token / profile | Deploy and manage resources |
| `TINKER_API_KEY` in `spindle-api` secret | Authenticate Tinker clients |
| `MODAL_PROXY_TOKEN_ID`/`MODAL_PROXY_TOKEN_SECRET` in `spindle-proxy` secret | Reach protected sampler pools |
| `HF_TOKEN` in `huggingface-secret` secret | Download base weights |

## Commands

- `uv run spindle deploy`: interactive picker; without a terminal and without args it exits, deploying nothing. Checks secrets first and prints creation commands for missing ones.
- `uv run spindle deploy <name...>`: bundled configs by name. `--all`: every bundled config (in `src/spindle/configs`).
- `uv run spindle deploy config.py`: custom config file.
- Each deploy adds or updates the listed configs in the per-frontend registry; configs not listed remain served. Plain model IDs select the longest-context ready config regardless of origin; exact recipe names win.

## Custom configs

```bash
spindle config init --preset qwen35-9b-lora-16k > my_model.py
spindle config validate my_model.py
spindle deploy my_model.py
```

Subclass `BaseConfig` or a bundled `Config`. Dotted `overrides` set single backend options (`{"miles_cfg.max_tokens_per_gpu": 8192}`). `trainer_gpu`/`inference_gpu` accept a preference list like `["H200", "B300"]`; Modal schedules the first with capacity. Platform settings (`platform.frontend`, `.modal.environment`, `.modal.region`, `.secrets.*`, `.storage.*`) must match across all configs in one deploy; `--app`/`--env`/`--region` flags override.

Fixed a bundled config? See the `spindle-dev` skill's upstream-config.md.

## Update isolation

- Apps: `spindle-trainer-{name}`, `spindle-inference-{name}`; pools `spindle-lora-{name}`, `spindle-fft-{digest}-*`. Recipe names must be unique per Modal environment.
- Inference-only change reuses the trainer app and vice versa. `--refresh-trainer`/`--refresh-inference` for source-only upgrades.
- Inference changes redeploy existing pools before the frontend publishes; an interrupted update retries with the same command. Region/storage/secret changes refresh both apps.
- Drain jobs before removing a recipe or changing model/context/adapter incompatibly.
- A startup failure pauses new starts for five minutes; `spindle deployment retry --frontend <app> <definition_id>` clears it.

## Teardown

Stopped heartbeats trigger a periodic cleaner that reclaims idle models and latest pools (`modal app list` to check). Manual: stop `spindle-fft-...` samplers first, then the frontend with `modal app stop`; stopping the frontend does not stop samplers.
