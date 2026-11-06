"""Nightly synthetic monitoring for spindle configs.

Test it out with:

```shell
    uv run modal run -e spindle-ci -m synthetic_monitoring.config_validation::launch_nightly
```
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import modal

from synthetic_monitoring.chart import RunPoint, render_timing_history_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
MODAL_ENV = "spindle-ci"
HISTORY_DICT_NAME = "spindle-synmon-prod-timings"
TRAINER_APPS = {
    "dapo-math": "spindle-trainer-prod-qwen38-27b-lora-64k",
    "swe-gym": "spindle-trainer-prod-qwen35-9b-lora-128k",
}
DAPO_DIR = "/opt/dapo-math"
SWE_GYM = "/opt/modal-examples/06_gpu_and_ml/swe_gym/swe_gym.py"
SWE_GYM_REV = "29c7c4929286d00903339ef863041592a99a2348"

SCRIPTS = {
    "dapo-math": {
        "clients": 2,
        "steps": 5,
        "timeout_s": 2 * 60 * 60,
    },
    "swe-gym": {
        "clients": 2,
        "steps": 5,
        "groups": 1,
        "timeout_s": 6 * 60 * 60,
    },
}
MONITOR_TIMEOUT_S = max(script["timeout_s"] for script in SCRIPTS.values()) + 15 * 60
LAUNCH_TIMEOUT_S = MONITOR_TIMEOUT_S + 30 * 60

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none --sparse"
        " https://github.com/modal-labs/modal-examples.git /opt/modal-examples"
        " && git -C /opt/modal-examples sparse-checkout set 06_gpu_and_ml/swe_gym"
        f" && git -C /opt/modal-examples checkout {SWE_GYM_REV}"
    )
    .run_commands(
        "sed -i 's/\"qwen35-9b-lora-128k\"/MODEL/'"
        " /opt/modal-examples/06_gpu_and_ml/swe_gym/swe_gym.py",
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
    .add_local_file(
        str(REPO_ROOT / "examples/shared-server/with_proxy_auth.py"),
        "/opt/proxy-auth/with_proxy_auth.py",
    )
    .add_local_file(
        str(REPO_ROOT / "synthetic_monitoring/sitecustomize.py"),
        "/opt/proxy-auth/sitecustomize.py",
    )
)

slack_secret = modal.Secret.from_name("spindle-bot-slack", environment_name=MODAL_ENV)
api_secret = modal.Secret.from_name("spindle-prod-api", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name=MODAL_ENV)

app = modal.App("spindle-synmon-launcher")


def run_script(name: str, base_url: str) -> dict[str, float]:
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
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, ["/opt/proxy-auth", env.get("PYTHONPATH")])
    )
    print("running", " ".join(command))
    started = time.time()
    subprocess.run(command, check=True, env=env)
    clients: dict[int | str, list[dict]] = {}
    for path in output.rglob("events.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            key = event.get("job", str(path))
            clients.setdefault(key, []).append(event)
    if len(clients) != spec["clients"]:
        raise ValueError(
            f"Expected events for {spec['clients']} clients, got {len(clients)}"
        )
    ready, first, last = [], [], []
    ready_event = "ready" if name == "dapo-math" else "rollout_start"
    step_event = "update_done" if name == "dapo-math" else "step_done"
    for events in clients.values():
        ready.append(next(e["time"] for e in events if e["event"] == ready_event))
        steps = sorted(e["time"] for e in events if e["event"] == step_event)
        if len(steps) != spec["steps"]:
            raise ValueError(f"Expected {spec['steps']} steps, got {len(steps)}")
        first.append(steps[0])
        last.append(steps[-1])
    return {
        "startup": max(ready) - started,
        "first step": max(first) - max(ready),
        "warm steps": max(last) - max(first),
    }


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
        delta = ", ".join(
            f"{stage}: {seconds - prior.timings[stage]:+.0f}s"
            for stage, seconds in point.timings.items()
        )
    return {
        "config": config,
        "status": point.status,
        "total_duration_s": point.total_duration_s if ok else None,
        "delta": delta,
        "timings": point.timings,
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
    started = time.time()
    error: str | None = None
    timings = {}
    try:
        timings = run_script(script, os.environ["TINKER_BASE_URL"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"error: {script} failed: {error}")

    try:
        url = modal.App.lookup(
            TRAINER_APPS[script], environment_name="spindle"
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
    rows: list[dict] = []
    started = time.time()
    calls = {script: monitor.spawn(script) for script in SCRIPTS}
    for script, call in calls.items():
        spec = SCRIPTS[script]
        timeout_s = spec["timeout_s"]
        try:
            rows.append(call.get(timeout=timeout_s))
        except Exception as exc:
            call.cancel()
            error = f"{type(exc).__name__}: {str(exc) or f'did not finish within {timeout_s}s'}"
            print(f"error: {script} failed: {error}")
            rows.append(_failed_row(script, started, error))
    _post_report(rows)
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError("synmon finished with failures")
    return rows
