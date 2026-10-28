"""Nightly synthetic monitoring for spindle configs.

Test it out with:

```shell
    uv run modal run -e ajhinh-dev -m synthetic_monitoring.config_validation::launch_nightly
```

TODO: change to shared env
"""

from __future__ import annotations

import contextlib
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import modal

from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig, config_path, load
from synthetic_monitoring.chart import RunPoint, render_timing_history_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
# TODO: change to shared env
MODAL_ENV = "ajhinh-dev"
HISTORY_DICT_NAME = "spindle-synmon-timing-baselines"
FRONTEND_NAME = "spindle-synmon"
WORKER_APP_PREFIXES = ("spindle-trainer-", "spindle-inference-", "spindle-lora-")

CONFIGS = {
    "qwen38-27b-lora-16k": 60 * 60,
    "qwen35-9b-lora-128k": 60 * 60,
    "qwen35-9b-lora-16k": 60 * 60,
    "deepseek-v41-flash-lora-8k": 2 * 60 * 60,
}
NUM_STEPS = 5
DEPLOY_TIMEOUT_S = 30 * 60
MONITOR_TIMEOUT_S = max(CONFIGS.values()) + 10 * 60
LAUNCH_TIMEOUT_S = DEPLOY_TIMEOUT_S + MONITOR_TIMEOUT_S + 30 * 60

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install("slack-sdk==3.27.1", "matplotlib==3.10.1")
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
    .add_local_python_source("spindle", "synthetic_monitoring")
)

slack_secret = modal.Secret.from_name("spindle-bot-slack", environment_name=MODAL_ENV)
api_secret = modal.Secret.from_name("spindle-api", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name=MODAL_ENV)

app = modal.App("spindle-synmon-launcher")


def synmon_configs(names=None):
    paths = [config_path(name) for name in names or CONFIGS]
    return [
        type(load(path))(
            overrides={
                "platform.frontend": FRONTEND_NAME,
                "platform.modal.environment": MODAL_ENV,
            }
        )
        for path in paths
    ]


def stop_worker_apps(names):
    for name in names:
        for prefix in WORKER_APP_PREFIXES:
            app_name = f"{prefix}{name}"
            try:
                worker = modal.App.lookup(app_name, environment_name=MODAL_ENV)
            except modal.exception.NotFoundError:
                continue
            print(f"stopping {app_name}")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "app",
                    "stop",
                    "-y",
                    worker.app_id,
                    "--env",
                    MODAL_ENV,
                ],
                check=True,
            )


def _history_dict():
    return modal.Dict.from_name(
        HISTORY_DICT_NAME, create_if_missing=True, environment_name=MODAL_ENV
    )


def _raw_cell(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def slack_report_blocks(rows: list[dict]) -> list[dict]:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    table_rows = [
        [
            _raw_cell("Config"),
            _raw_cell("Total runtime"),
            _raw_cell("Delta vs previous successful run"),
        ]
    ]
    for row in rows:
        if row["status"] == "success":
            runtime, delta = (
                _raw_cell(str(timedelta(seconds=round(row["total_duration_s"])))),
                _raw_cell(row["delta"]),
            )
        else:
            runtime, delta = _raw_cell("FAILED"), _raw_cell("n/a")
        table_rows.append([_raw_cell(row["config"]), runtime, delta])
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Nightly config validation - {day}",
            },
        },
        {
            "type": "table",
            "column_settings": [
                {"is_wrapped": True},
                {"align": "left"},
                {"is_wrapped": True},
            ],
            "rows": table_rows,
        },
    ]


def _record(config: str, point: RunPoint) -> list[RunPoint]:
    history = _history_dict()
    points = [RunPoint(**item) for item in history.get(config, [])]
    points.append(point)
    history[config] = [asdict(p) for p in points]
    return points


def _row(
    config: str, point: RunPoint, history: list[RunPoint], error: str | None = None
) -> dict:
    ok = point.status == "success"
    prior = next((p for p in reversed(history[:-1]) if p.status == "success"), None)
    delta = "n/a"
    if ok and prior:
        delta_s = point.total_duration_s - prior.total_duration_s
        delta = f"{delta_s:+.0f}s ({delta_s / prior.total_duration_s * 100:+.1f}%)"
    return {
        "config": config,
        "status": point.status,
        "total_duration_s": point.total_duration_s if ok else None,
        "delta": delta,
        "modal_app_url": point.modal_app_url,
        "error": error,
    }


def _post_report(rows: list[dict]) -> None:
    from slack_sdk import WebClient

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    channel = os.environ["SLACK_CHANNEL_ID"]

    header = f"Nightly config validation - {datetime.now(timezone.utc):%Y-%m-%d}"
    thread_ts = client.chat_postMessage(
        channel=channel,
        text=header,
        blocks=slack_report_blocks(rows),
        mrkdwn=True,
        unfurl_links=False,
        unfurl_media=False,
    )["ts"]
    for row in rows:
        try:
            if row["status"] == "success":
                png = render_timing_history_chart(
                    [
                        RunPoint(**item)
                        for item in _history_dict().get(row["config"], [])
                    ],
                    config_name=row["config"],
                )
                if png:
                    client.files_upload_v2(
                        channel=channel,
                        thread_ts=thread_ts,
                        content=png,
                        filename=f"timing_history_{row['config']}.png",
                        title=row["config"],
                    )
                continue
            text = f"`{row['config']}` failed"
            if row.get("error"):
                text += f"\n```{row['error'][:1500]}```"
            if row.get("modal_app_url"):
                text += f"\n<{row['modal_app_url']}|Modal logs>"
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=text,
                mrkdwn=True,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            print(f"error: reply failed for {row['config']}: {exc}")


@contextlib.contextmanager
def _stage(timings: dict[str, float], name: str):
    start = time.time()
    yield
    timings[name] = time.time() - start


@app.function(
    image=probe_image,
    timeout=MONITOR_TIMEOUT_S,
    secrets=[api_secret, hf_secret],
)
def monitor(config: str) -> dict:
    import tinker
    from tinker import types

    recipe = synmon_configs(names=[config])[0]
    started = time.time()
    timings: dict[str, float] = {}
    error: str | None = None
    try:
        with _stage(timings, "trainer ready"):
            service = tinker.ServiceClient(
                base_url=modal.Function.from_name(
                    FRONTEND_NAME, "server", environment_name=MODAL_ENV
                ).get_web_url(),
                api_key=os.environ["TINKER_API_KEY"],
            )
            training = service.create_lora_training_client(base_model=recipe.name)
            tokenizer = training.get_tokenizer()
        prompt = tokenizer.encode(
            "What is 2 + 2? Answer with only the number.", add_special_tokens=True
        )

        def sample(sampling) -> tuple[list[int], list[float]]:
            sampled = sampling.sample(
                prompt=types.ModelInput.from_ints(prompt),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=16, temperature=1.0),
            ).result()
            sequence = sampled.sequences[0]
            tokens = list(sequence.tokens)
            logprobs = list(sequence.logprobs or ())
            if (
                not tokens
                or len(logprobs) != len(tokens)
                or not all(map(math.isfinite, logprobs))
            ):
                raise RuntimeError("sampling returned invalid tokens or logprobs")
            return tokens, logprobs

        first_step_start = time.time()
        sampling = training.save_weights_and_get_sampling_client()
        step_timings: list[dict[str, float]] = []
        for step in range(NUM_STEPS):
            s: dict[str, float] = {}
            with _stage(s, "sample"):
                response_tokens, response_logprobs = sample(sampling)
            response_text = tokenizer.decode(response_tokens)
            reward = 1.0 if "4" in response_text else -1.0
            prompt_targets = len(prompt) - 1
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(prompt + response_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": [0] * prompt_targets + response_tokens,
                    "logprobs": [0.0] * prompt_targets + response_logprobs,
                    "advantages": [0.0] * prompt_targets
                    + [reward] * len(response_tokens),
                },
            )
            with _stage(s, "forward_backward"):
                training.forward_backward([datum], "importance_sampling").result()
            with _stage(s, "optim_step"):
                optim = training.optim_step(
                    types.AdamParams(learning_rate=1e-5)
                ).result()
                if optim.metrics["update_successful:mean"] != 1.0:
                    raise RuntimeError(
                        f"optim_step skipped the update: {optim.metrics}"
                    )
            with _stage(s, "save weights"):
                sampling = training.save_weights_and_get_sampling_client()
            step_timings.append(s)
            if step == 0:
                timings["first step"] = time.time() - first_step_start
            else:
                for name in s:
                    timings[name] = statistics.mean(d[name] for d in step_timings[1:])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"error: probe failed for {config}: {error}")

    try:
        url = modal.App.lookup(
            f"spindle-trainer-{recipe.name}", environment_name=MODAL_ENV
        ).get_dashboard_url()
    except Exception:
        url = None
    point = RunPoint(
        ts=started,
        timings=timings,
        total_duration_s=time.time() - started,
        status="failed" if error else "success",
        modal_app_url=url,
    )
    return _row(config, point, _record(config, point), error)


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("0 2 * * *", timezone="America/Los_Angeles"),
    secrets=[slack_secret],
)
def launch_nightly() -> list[dict]:
    recipes = synmon_configs()
    names = [r.name for r in recipes]
    stop_worker_apps(names)
    deploy([DeploymentConfig.create(recipe) for recipe in recipes])
    rows: list[dict] = []
    try:
        calls = {name: monitor.spawn(name) for name in CONFIGS}
        start = time.time()
        for name, call in calls.items():
            try:
                rows.append(
                    call.get(timeout=max(0, start + CONFIGS[name] - time.time()))
                )
            except Exception as exc:
                call.cancel()
                error = f"{type(exc).__name__}: {str(exc) or f'did not finish within {CONFIGS[name]}s'}"
                print(f"error: probe failed for {name}: {error}")
                point = RunPoint(
                    ts=start,
                    timings={},
                    total_duration_s=time.time() - start,
                    status="failed",
                )
                rows.append(_row(name, point, _record(name, point), error))
    finally:
        stop_worker_apps(names)

    _post_report(rows)
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError("synmon fan-out finished with failures")
    return rows
