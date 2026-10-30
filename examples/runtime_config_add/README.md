# Runtime config-add smoke test

`smoke_base.py` is a standalone recipe, not a bundled config, so it is not
included in `spindle deploy --all`. It uses the Qwen3.5 2B LoRA recipe with a
4k context, one H100 for training and inference, and one trainer client.

Validate and deploy the base recipe to the dev workspace:

```bash
spindle config validate examples/runtime_config_add/smoke_base.py
spindle deploy examples/runtime_config_add/smoke_base.py
```

For adding runtime variants, see [Deployment configs](../../docs/deployment-configs.md).

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
