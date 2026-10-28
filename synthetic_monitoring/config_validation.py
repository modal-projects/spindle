"""Nightly synthetic monitoring for spindle configs.

Test it out with:

```shell
    uv run modal run -e spindle-ci -m synthetic_monitoring.config_validation::launch_nightly
```
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import modal

from spindle.deployment_cli import deploy
from spindle.deployments import DeploymentConfig, config_path, load
from synthetic_monitoring.chart import RunPoint, render_timing_history_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
MODAL_ENV = "spindle-ci"
HISTORY_DICT_NAME = "spindle-synmon-timing-baselines"
FRONTEND_NAME = "spindle-synmon"
WORKER_APP_PREFIXES = ("spindle-trainer-", "spindle-inference-", "spindle-lora-")
DAPO_DIR = "/opt/dapo-math"
SWE_GYM = "/opt/modal-examples/06_gpu_and_ml/swe_gym/swe_gym.py"
SWE_GYM_REV = "29c7c4929286d00903339ef863041592a99a2348"

CONFIGS = (
    "qwen38-27b-lora-16k",
    "qwen35-9b-lora-128k",
    "qwen35-9b-lora-16k",
    "deepseek-v41-flash-lora-8k",
)
GPU_BUDGET = {
    "trainer_gpu": "H200",
    "trainer_gpus_per_node": 2,
    "trainer_nodes": 1,
    "trainer_max_instances": 1,
    "trainer_max_clients_per_instance": 2,
    "inference_gpu": "B200",
    "inference_gpus_per_node": 1,
    "inference_min_replicas": 1,
    "inference_max_replicas": 2,
    "miles_cfg.max_lora_slots": 2,
    "sglang_cfg.tp_size": 1,
    "sglang_cfg.max_running_requests": 4,
}
SWE_TRAINER = {
    "max_context_length": 131_072,
    "trainer_gpus_per_node": 4,
    "miles_cfg.tensor_model_parallel_size": 1,
    "miles_cfg.context_parallel_size": 4,
    "miles_cfg.max_tokens_per_gpu": 32768,
    "miles_cfg.align_sequences_to_parallel_layout": True,
    "miles_cfg.cli_options.cp_comm_type": ["p2p"],
    "sglang_cfg.chunked_prefill_size": 8192,
    "sglang_cfg.mem_fraction_static": 0.85,
}
SCRIPTS = {
    "dapo-math": {
        "clients": 2,
        "steps": 5,
        "timeout_s": 2 * 60 * 60,
        "recipe": "qwen38-27b-lora-16k",
        "overrides": {
            "max_context_length": 8192,
            "trainer_gpus_per_node": 2,
            "miles_cfg.tensor_model_parallel_size": 2,
            "miles_cfg.max_tokens_per_gpu": 8192,
        },
    },
    "swe-gym": {
        "clients": 2,
        "steps": 5,
        "groups": 1,
        "timeout_s": 6 * 60 * 60,
        "recipe": "qwen35-9b-lora-128k",
        "overrides": SWE_TRAINER,
    },
}
DEPLOY_TIMEOUT_S = 30 * 60
MONITOR_TIMEOUT_S = max(script["timeout_s"] for script in SCRIPTS.values()) + 15 * 60
LAUNCH_TIMEOUT_S = DEPLOY_TIMEOUT_S + MONITOR_TIMEOUT_S + 30 * 60

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none --sparse"
        " https://github.com/modal-labs/modal-examples.git /opt/modal-examples"
        " && git -C /opt/modal-examples sparse-checkout set 06_gpu_and_ml/swe_gym"
        f" && git -C /opt/modal-examples checkout {SWE_GYM_REV}"
    )
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install(
        "slack-sdk==3.27.1",
        "matplotlib==3.10.1",
        "transformers[chat-template]==5.17.0",
        "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7",
    )
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
    .add_local_python_source("spindle", "synthetic_monitoring")
    .add_local_dir(
        str(REPO_ROOT / "examples/dapo-27b-4h200-12b200"),
        remote_path=DAPO_DIR,
    )
)

slack_secret = modal.Secret.from_name("spindle-bot-slack", environment_name=MODAL_ENV)
api_secret = modal.Secret.from_name("spindle-api", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name=MODAL_ENV)

app = modal.App("spindle-synmon-launcher")


def synmon_configs(names=None, extra_overrides=None):
    extra = extra_overrides or {}
    recipes = []
    for name in names or CONFIGS:
        recipe = load(config_path(name))
        recipes.append(
            type(recipe)(
                overrides={
                    "platform.frontend": FRONTEND_NAME,
                    "platform.modal.environment": MODAL_ENV,
                    **GPU_BUDGET,
                    **extra,
                }
            )
        )
    return recipes


def run_script(name: str, base_url: str) -> None:
    spec = SCRIPTS[name]
    output = Path(tempfile.mkdtemp(prefix=f"synmon-{name}-"))
    if name == "dapo-math":
        work = output / "dapo"
        shutil.copytree(DAPO_DIR, work)
        cfg = json.loads((work / "config.json").read_text())
        cfg.update(
            updates=spec["steps"],
            context_tokens=8192,
            max_tokens=1024,
            groups_per_update=1,
            samples_per_group=4,
            max_candidate_groups=4,
            warmup_samples=1,
        )
        config = work / "config.json"
        config.write_text(json.dumps(cfg))
        command = [
            sys.executable,
            str(work / "launch_clients.py"),
            "--clients",
            str(spec["clients"]),
            "--mode",
            "gate",
            "--stagger-seconds",
            "10",
            "--config",
            str(config),
            "--base-url",
            base_url,
        ]
    else:
        command = [
            sys.executable,
            SWE_GYM,
            "--clients",
            str(spec["clients"]),
            "--steps",
            str(spec["steps"]),
            "--groups-per-batch",
            str(spec["groups"]),
            "--hours",
            str(spec["timeout_s"] / 3600),
            "--output",
            str(output),
        ]
    env = os.environ.copy()
    env["TINKER_BASE_URL"] = base_url
    print("running", " ".join(command))
    subprocess.run(command, check=True, env=env)


def stop_worker_apps(names) -> list[str]:
    failures = []
    for name in names:
        for prefix in WORKER_APP_PREFIXES:
            app_name = f"{prefix}{name}"
            try:
                worker = modal.App.lookup(app_name, environment_name=MODAL_ENV)
            except modal.exception.NotFoundError:
                continue
            except Exception as exc:
                message = f"lookup {app_name} failed: {exc}"
                print(f"error: {message}")
                failures.append(message)
                continue
            print(f"stopping {app_name}")
            result = subprocess.run(
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
                check=False,
            )
            if result.returncode:
                message = f"stopping {app_name} failed"
                print(f"error: {message}")
                failures.append(message)
    return failures


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
            _raw_cell("Script"),
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
                "text": f"Nightly validation - {day}",
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

    header = f"Nightly validation - {datetime.now(timezone.utc):%Y-%m-%d}"
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


@app.function(
    image=probe_image,
    timeout=MONITOR_TIMEOUT_S,
    secrets=[api_secret, hf_secret],
)
def monitor(script: str) -> dict:
    recipe = SCRIPTS[script]["recipe"]
    started = time.time()
    error: str | None = None
    try:
        run_script(
            script,
            modal.Function.from_name(
                FRONTEND_NAME, "server", environment_name=MODAL_ENV
            ).get_web_url(),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"error: {script} failed: {error}")

    try:
        url = modal.App.lookup(
            f"spindle-trainer-{recipe}", environment_name=MODAL_ENV
        ).get_dashboard_url()
    except Exception:
        url = None
    point = RunPoint(
        ts=started,
        timings={},
        total_duration_s=time.time() - started,
        status="failed" if error else "success",
        modal_app_url=url,
    )
    return _row(script, point, _record(script, point), error)


def _failed_row(script: str, started: float, error: str) -> dict:
    point = RunPoint(
        ts=started,
        timings={},
        total_duration_s=max(0, time.time() - started),
        status="failed",
    )
    try:
        history = _record(script, point)
    except Exception as exc:
        print(f"error: recording {script} failed: {exc}")
        history = [point]
    return _row(script, point, history, error)


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("0 2 * * *", timezone="America/Los_Angeles"),
    secrets=[slack_secret],
)
def launch_nightly() -> list[dict]:
    script = random.choice(tuple(SCRIPTS))
    spec = SCRIPTS[script]
    print(f"synmon script: {script}")
    rows: list[dict] = []
    started = time.time()
    try:
        recipes = synmon_configs([spec["recipe"]], spec.get("overrides"))
        stop_worker_apps([spec["recipe"]])
        deploy([DeploymentConfig.create(recipe) for recipe in recipes])
        call = monitor.spawn(script)
        timeout_s = spec["timeout_s"]
        try:
            rows.append(call.get(timeout=timeout_s))
        except Exception as exc:
            call.cancel()
            error = f"{type(exc).__name__}: {str(exc) or f'did not finish within {timeout_s}s'}"
            print(f"error: {script} failed: {error}")
            rows.append(_failed_row(script, started, error))
    except Exception as exc:
        if not rows:
            error = f"{type(exc).__name__}: {exc}"
            print(f"error: {script} failed: {error}")
            rows.append(_failed_row(script, started, error))
    finally:
        cleanup = stop_worker_apps([spec["recipe"]])

    if cleanup:
        message = "cleanup failed: " + "; ".join(cleanup)
        print(f"error: {message}")
        if rows and rows[-1]["status"] != "success":
            prior = rows[-1].get("error") or ""
            rows[-1]["error"] = f"{prior}\n{message}".strip()
        elif rows:
            rows[-1]["status"] = "failed"
            rows[-1]["delta"] = "n/a"
            rows[-1]["total_duration_s"] = None
            rows[-1]["error"] = message
        else:
            rows.append(_failed_row(script, started, message))

    _post_report(rows)
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError("synmon finished with failures")
    return rows
