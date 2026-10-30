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
- `uv run spindle deploy --add NAME --base BASE [--FIELD=VALUE | --FIELD VALUE ...]`: add a config to a running frontend. Pass recipe fields directly; dotted fields are supported, and values are parsed as JSON with a raw-string fallback. For example:

  ```bash
  uv run spindle deploy --add qwen35-9b-lora-16k-h200 \
    --base qwen35-9b-lora-16k \
    --trainer_gpu=H200 \
    --miles_cfg.max_tokens_per_gpu=8192
  ```

- `uv run spindle deploy --remove NAME`: remove a config from a running frontend (`DELETE /api/v1/spindle/deployments/NAME`); it stops routing on the frontend's next registry read.
- The equivalent API request is `POST /api/v1/spindle/deployments`, with a JSON body containing `name`, a ready `base`, and `overrides`. Poll `GET /api/v1/spindle/deployments` until the row is `ready` or reports an error. `DELETE /api/v1/spindle/deployments/{name}` removes a config from the frontend (HTTP 404 if the name is unknown):

  ```bash
  curl -sS -X POST -H "X-API-Key: $TINKER_API_KEY" \
    -H 'Content-Type: application/json' \
    "$TINKER_BASE_URL/api/v1/spindle/deployments" \
    -d '{"name":"qwen35-9b-lora-16k-h200","base":"qwen35-9b-lora-16k","overrides":{"trainer_gpu":"H200","miles_cfg.max_tokens_per_gpu":8192}}'
  curl -sS -H "X-API-Key: $TINKER_API_KEY" \
    "$TINKER_BASE_URL/api/v1/spindle/deployments"
  ```

- Invalid config posts return HTTP 400; reusing a name with different settings returns HTTP 409.
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
