# Shared server

One Spindle deployment that a whole team points tinker-cookbook at, instead of each person deploying their own. It is what the Spindle maintainers run for themselves: recipes on a frontend named `prod` in a Modal environment named `spindle`, which holds nothing else.

Each file in `recipes/` subclasses a bundled recipe and changes its name to start with `prod-`, so its apps are easy to tell apart, and points it at the `prod` frontend in the `spindle` environment. The frontend sets `platform.url_label` to `prod`, so its URL is `https://<workspace>-spindle--prod.modal.run`. It also sets `platform.requires_proxy_auth`, so Modal turns away any request without a proxy token before it reaches Spindle.

`recipes/` holds one LoRA recipe per base model, 20 in all, from Qwen3.5-2B to Nemotron-3-Ultra-550B.

## Deploy

Create the environment and its `spindle-api`, `spindle-proxy` and `huggingface-secret` secrets as the [README](../../README.md#secrets) describes. Then deploy all the recipes. A deploy adds or updates only the recipes you list; recipes already on the server keep serving:

```bash
uv run spindle deploy examples/shared-server/recipes/*.py
```

A redeploy with the same Spindle version only republishes the frontend, and every run keeps going: state lives in Modal Dicts named after the frontend's app ID, and a redeploy keeps that ID. A new Spindle version rebuilds every trainer and sampler app, so schedule it with users. Stopping the frontend app gives the next deploy a new app ID and loses that state, so update it with `spindle deploy`, never with `modal app stop`.

## Use

Clients need the URL, the Tinker API key, and a Modal [proxy token](https://modal.com/docs/guide/webhook-proxy-auth). Anyone in the workspace can create their own token and allow it for the environment:

```bash
modal workspace proxy-tokens create          # prints the token ID (wk-...) and secret (ws-...) once
modal workspace proxy-tokens allow <wk-token-id> spindle
```

Then set:

```bash
export TINKER_BASE_URL=https://<workspace>-spindle--prod.modal.run
export TINKER_API_KEY=tml-...
export MODAL_PROXY_TOKEN_ID=wk-...
export MODAL_PROXY_TOKEN_SECRET=ws-...
```

The Tinker SDK sends the API key on its own but has no setting for the proxy token. Run scripts through [`with_proxy_auth.py`](with_proxy_auth.py), which adds the token to every client the script creates. It only needs `tinker`, so you can copy it next to your own code. tinker-cookbook commands work unchanged:

```bash
python examples/shared-server/with_proxy_auth.py -m tinker_cookbook.recipes.chat_sl.train \
    model_name=Qwen/Qwen3.8-27B base_url=$TINKER_BASE_URL
```

In your own code you can pass the headers yourself instead. Tinker 0.36 and newer call the argument `default_headers_overrides`; older versions call it `default_headers`:

```python
service = tinker.ServiceClient(
    default_headers_overrides={
        "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
    }
)
```

The server picks the recipe from the client's `base_model`.

## Check

`smoke.py` trains two LoRA steps on one recipe and, with `--expect-commit`, first waits until the server reports that commit:

```bash
uv run python examples/shared-server/smoke.py --base-model Qwen/Qwen3-4B --expect-commit "$(git rev-parse HEAD)"
```

The workflow in `.github/workflows/deploy-shared-server.yml` runs the deploy and this check after every green test run on `main`.
