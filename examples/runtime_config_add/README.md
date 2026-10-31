# Runtime config-add smoke test

`smoke_base.py` is a standalone recipe, not a bundled config, so it is not
included in `spindle deploy --all`. It uses the Qwen3.5 2B LoRA recipe with a
4k context, one H100 for training and inference, and one trainer client.

Validate and deploy the base recipe to the dev workspace:

```bash
spindle config validate examples/runtime_config_add/smoke_base.py
spindle deploy examples/runtime_config_add/smoke_base.py
```

Set the frontend URL and API key, then add both runtime variants without
redeploying the frontend:

```bash
export TINKER_BASE_URL=... TINKER_API_KEY=...

uv run spindle deploy --add runtime-add-smoke-2b-8k \
  --base runtime-add-smoke-2b \
  --max_context_length=8192 \
  --miles_cfg.max_tokens_per_gpu=8192

uv run spindle deploy --add runtime-add-smoke-2b-kill \
  --base runtime-add-smoke-2b \
  --inference_scaledown_window_s=120

curl -sS -i -X POST "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"runtime-add-smoke-2b-8k","base":"runtime-add-smoke-2b","overrides":{"max_context_length":8192,"miles_cfg.max_tokens_per_gpu":8192}}'

curl -sS -i -X POST "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"runtime-add-smoke-2b-kill","base":"runtime-add-smoke-2b","overrides":{"inference_scaledown_window_s":120}}'

curl -sS -i "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY"
```

The CLI accepts the same fields directly, parses JSON values with a raw-string
fallback, and waits until each deployment is ready or fails. The curl commands
submit the same API requests and can be polled with the deployments GET.

Each new POST returns HTTP 202 with `state=deploying`; the GET returns HTTP 200.
Poll the GET until each added row is `ready`. An invalid config returns HTTP
400:

```bash
curl -sS -i -X POST "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"runtime-add-smoke-invalid","base":"runtime-add-smoke-2b","overrides":{"max_context_length":"big"}}'
```

A repeated identical POST is idempotent and returns HTTP 202. Reusing the 8k
name with different settings returns HTTP 409:

```bash
curl -sS -i -X POST "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"runtime-add-smoke-2b-8k","base":"runtime-add-smoke-2b","overrides":{"max_context_length":4096}}'
```

The 8k variant is intentionally longer than the base. A plain
`Qwen/Qwen3.5-2B` request should still select the 4k base recipe; use the
config name to select either added variant.

With `TINKER_BASE_URL` and `TINKER_API_KEY` set, run the shared-server smoke
test through the proxy-auth wrapper:

```bash
uv run --frozen python examples/shared-server/with_proxy_auth.py \
  examples/shared-server/smoke.py --base-model <recipe name or model id>
```

## Cancel an in-flight added deployment

The frontend spawns a `deploy_recipe` function call and stores its Modal call ID
in the deployment row. The deployments API does not return that ID. In the
Modal dashboard, find the active `deploy_recipe` call in the frontend app and
copy its call ID. Cancel it from a machine with the Modal credentials:

```bash
uv run --frozen python -c \
  'import modal; modal.FunctionCall.from_id("fc-...").cancel(terminate_containers=True)'
```

Cancellation terminates the function container, so its error handler may not
write a failure state. The row can remain `deploying` until its `updated_at` is
older than 65 minutes; then GET `/api/v1/spindle/deployments` reports
`failed` with `deploy worker stopped without reporting a result`. Posting the
same name and settings after it reports failed retries the deployment.
If a trainer, inference, or rollout worker deploy command exits with an error instead,
the row fails immediately with an error prefixed by `trainer app deploy failed:`
`inference app deploy failed:`, or `rollout app deploy failed:`.
