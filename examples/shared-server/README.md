# Shared server

One Spindle deployment that a whole team points tinker-cookbook at, instead of each person deploying their own. It serves four recipes on a frontend named `spindle-shared`, and it is what the Spindle maintainers run for themselves.

Each file here subclasses a bundled recipe and changes three things: the recipe gets a `shared-` name so it never collides with someone else's deployment of the same recipe in the same environment, the frontend is `spindle-shared`, and the API key and checkpoint volume are the deployment's own.

| Recipe | Base model | Trainer |
|---|---|---|
| `shared-qwen35-2b-lora-32k` | Qwen/Qwen3.5-2B | 4× H100 |
| `shared-qwen38-27b-lora-64k` | Qwen/Qwen3.8-27B | 8× H200 |
| `shared-qwen36-35b-a3b-lora-32k` | Qwen/Qwen3.6-35B-A3B | 8× H200 |
| `shared-deepseek-v41-flash-lora-32k` | deepseek-ai/DeepSeek-V4.1-Flash | 2 × 8 B300 |

## Deploy

Create the API-key secret once in the target environment, next to the `spindle-proxy` and `huggingface-secret` secrets from the [README](../../README.md#secrets):

```bash
modal secret create spindle-shared-api TINKER_API_KEY="tml-$(openssl rand -hex 32)" --env <env>
```

Then deploy all four recipes together. Every deploy must list all of them, because the argument list is the full set the frontend serves:

```bash
uv run spindle deploy examples/shared-server/*.py --env <env>
```

Redeploying keeps every running job: state lives in Modal Dicts named after the frontend's app ID, and a redeploy keeps that ID. Stopping the frontend app does not, so update it with `spindle deploy`, never with `modal app stop`.

## Use

Clients need only the URL and the key:

```bash
export TINKER_BASE_URL=https://<workspace>-<env>--spindle-shared-server.modal.run
export TINKER_API_KEY=tml-...
```

Any stock tinker or tinker-cookbook script then works unchanged. The server picks the recipe from the script's `base_model`.

## Check

`smoke.py` trains two LoRA steps on one recipe and, with `--expect-commit`, first waits until the server reports that commit:

```bash
uv run python examples/shared-server/smoke.py --base-model Qwen/Qwen3.5-2B --expect-commit "$(git rev-parse HEAD)"
```

The workflow in `.github/workflows/deploy-shared-server.yml` runs the deploy and this check after every green test run on `main`.
