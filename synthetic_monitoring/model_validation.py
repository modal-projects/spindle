"""Weekly synthetic monitoring for spindle model servers.

Probes every model the shared deployment serves (client-side one-step RL
run against its own parameterization) plus each registered FFT engine inside
an ephemeral `spindle.run` scope, then posts a Slack table + per-probe
timing-history charts.

Test it out with:

```shell
    uv run modal run -e ajhinh-dev -m synthetic_monitoring.model_validation --name lora-shared/qwen36-35b-a3b-lora-32k
```
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import modal

from synthetic_monitoring.chart import RunPoint, render_timing_history_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
# TODO: swap to the central team deployment env once it exists.
MODAL_ENV = "ajhinh-dev"
HISTORY_DICT_NAME = "spindle-synmon-baselines"
SHARED_FRONTEND = "spindle"
# TODO: move to #proj-train-spindle (C0BMATL1S4F) once reports go public.
REPORT_CHANNEL_ID = "C0CDETGBHC0"
SAMPLE_PROMPT = "What is 2 + 2? Answer with only the number."
CALL_TIMEOUT_S = 60 * 60
CLEANUP_GRACE_S = 5 * 60
FRONTEND_TIMEOUT_S = 2 * 60

# TODO: extend as new engines are registered in spindle.engines.
FFT_ENGINES = (
    "qwen3_5_4b_full_64k",
    "qwen3_6_27b_full_64k",
    "qwen3_5_9b_full_64k",
)


@dataclass(frozen=True)
class _ProbeKind:
    name: str
    timeout_s: int


PROBE_KINDS = (
    _ProbeKind("lora-shared", 3 * 60 * 60),
    _ProbeKind("fft-scoped", 3 * 60 * 60),
)
PROBE_TIMEOUT_S = max(c.timeout_s for c in PROBE_KINDS)
MONITOR_TIMEOUT_S = PROBE_TIMEOUT_S + CLEANUP_GRACE_S + 30 * 60
LAUNCH_TIMEOUT_S = MONITOR_TIMEOUT_S + 30 * 60

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install("slack-sdk==3.27.1", "matplotlib==3.10.1")
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
    .add_local_python_source("spindle", "synthetic_monitoring")
)

slack_secret = modal.Secret.from_name("gym-bot-slack", environment_name=MODAL_ENV)
api_secret = modal.Secret.from_name("spindle-api", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name=MODAL_ENV)

app = modal.App("spindle-synmon")


def _history_dict():
    return modal.Dict.from_name(
        HISTORY_DICT_NAME, create_if_missing=True, environment_name=MODAL_ENV
    )


def _record(probe: str, point: RunPoint) -> list[RunPoint]:
    try:
        history = _history_dict()
        points = [RunPoint(**item) for item in history.get(probe, [])]
        points.append(point)
        points.sort(key=lambda p: p.ts)
        history[probe] = [asdict(p) for p in points]
        return points
    except Exception as exc:
        print(f"error: append_history failed for {probe}: {exc}")
        return [point]


def _frontend_function(fn_name: str):
    return modal.Function.from_name(
        SHARED_FRONTEND, fn_name, environment_name=MODAL_ENV
    )


def _served_configs() -> list[dict]:
    """Config rows the shared frontend currently serves."""
    call = _frontend_function("deployed_configs").spawn()
    return call.get(timeout=FRONTEND_TIMEOUT_S)


def _raw_cell(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def _format_duration(total_s: float) -> str:
    total = int(round(total_s))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def slack_report_blocks(
    rows: list[dict], *, when: datetime | None = None
) -> list[dict]:
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    table_rows = [
        [
            _raw_cell("Probe"),
            _raw_cell("Total runtime"),
            _raw_cell("Delta vs previous successful run"),
        ]
    ]
    for row in rows:
        if row["status"] == "success":
            runtime = _raw_cell(_format_duration(row["total_duration_s"]))
            delta = _raw_cell(row["delta"])
        elif row.get("url"):
            runtime = _raw_cell("FAILED")
            delta = {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "link", "text": "server", "url": row["url"]}
                        ],
                    }
                ],
            }
        else:
            runtime = _raw_cell("FAILED")
            delta = _raw_cell((row.get("error") or "failed")[:200])
        table_rows.append([_raw_cell(row["probe"]), runtime, delta])
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Weekly spindle probe - {day}"},
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


def _format_error(exc: BaseException, subject: str, timeout_s: int) -> str:
    message = str(exc)
    if not message and isinstance(exc, TimeoutError):
        message = f"{subject} did not finish within {timeout_s}s"
    return f"{type(exc).__name__}: {message}"


def _row(
    probe: str, point: RunPoint, history: list[RunPoint], error: str | None = None
) -> dict:
    prior = next(
        (p for p in reversed(history) if p.status == "success" and p.ts != point.ts),
        None,
    )
    ok = point.status == "success"
    if ok:
        prior_s = prior.total_duration_s if prior else None
        if prior_s is None:
            delta = "n/a"
        else:
            delta_s = point.total_duration_s - prior_s
            delta = (
                f"{delta_s:+.0f}s"
                if prior_s <= 0
                else f"{delta_s:+.0f}s ({delta_s / prior_s * 100:+.1f}%)"
            )
    else:
        delta = "n/a"
    return {
        "probe": probe,
        "status": point.status,
        "total_duration_s": point.total_duration_s if ok else None,
        "delta": delta,
        "url": point.url,
        "timings": point.timings,
        "error": error,
    }


def _post_report(rows: list[dict], channel: str | None = None) -> None:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    channel = channel or os.environ["SLACK_CHANNEL_ID"]
    try:
        client.conversations_join(channel=channel)
    except SlackApiError:
        pass
    header = f"Weekly spindle probe - {datetime.now(timezone.utc):%Y-%m-%d}"
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
                        for item in _history_dict().get(row["probe"], [])
                    ],
                    probe_name=row["probe"],
                )
                if png:
                    try:
                        client.files_upload_v2(
                            channel=channel,
                            thread_ts=thread_ts,
                            content=png,
                            filename=f"timing_history_{row['probe'].replace('/', '_')}.png",
                            title=row["probe"],
                        )
                        continue
                    except SlackApiError as exc:
                        print(f"error: chart upload failed for {row['probe']}: {exc}")
                lines = [f"`{row['probe']}` phase timings:"]
                for phase, duration_s in row["timings"].items():
                    lines.append(f"  {phase}: {_format_duration(duration_s)}")
                text = "\n".join(lines)
            else:
                text = f"`{row['probe']}` failed"
                if row.get("error"):
                    text += f"\n```{row['error'][:1500]}```"
                if row.get("url"):
                    text += f"\n<{row['url']}|server>"
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=text,
                mrkdwn=True,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            print(f"error: reply failed for {row['probe']}: {exc}")


def _run_rl_step(
    base_url: str, api_key: str, model: str, parameterization: str
) -> dict[str, float]:
    import tinker
    from tinker import types

    timings: dict[str, float] = {}

    t = time.time()
    service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
    if parameterization == "lora":
        training = service.create_lora_training_client(base_model=model, rank=16)
    else:
        from spindle.client import create_full_training_client

        training = create_full_training_client(service, model)
    tokenizer = training.get_tokenizer()
    prompt_tokens = tokenizer.encode(SAMPLE_PROMPT, add_special_tokens=True)
    prompt = types.ModelInput.from_ints(prompt_tokens)
    params = types.SamplingParams(max_tokens=16, temperature=1.0)
    timings["client_setup"] = time.time() - t

    t = time.time()
    sampling = training.save_weights_and_get_sampling_client()
    first = (
        sampling.sample(prompt=prompt, num_samples=1, sampling_params=params)
        .result(timeout=CALL_TIMEOUT_S)
        .sequences[0]
    )
    tokens = list(first.tokens)
    logprobs = list(first.logprobs or ())
    if not tokens or len(logprobs) != len(tokens):
        raise RuntimeError("sampling returned invalid tokens or logprobs")
    timings["first_sample"] = time.time() - t

    t = time.time()
    text = tokenizer.decode(tokens)
    reward = 1.0 if "4" in text else -1.0
    prompt_targets = len(prompt_tokens) - 1
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(prompt_tokens + tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": prompt_tokens[1:] + tokens,
            "logprobs": [0.0] * prompt_targets + logprobs,
            "advantages": [0.0] * prompt_targets + [reward] * len(tokens),
        },
    )
    forward = training.forward_backward([datum], "importance_sampling")
    optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))
    forward.result(timeout=CALL_TIMEOUT_S)
    optimizer.result(timeout=CALL_TIMEOUT_S)
    timings["train_step"] = time.time() - t

    t = time.time()
    updated = training.save_weights_and_get_sampling_client()
    (
        updated.sample(prompt=prompt, num_samples=1, sampling_params=params).result(
            timeout=CALL_TIMEOUT_S
        )
    )
    timings["publish_and_resample"] = time.time() - t
    return timings


def _probe_lora_shared(config_name: str) -> tuple[dict[str, float], str, str]:
    recipe = next(
        (
            row["recipe"]
            for row in _served_configs()
            if row["recipe"]["name"] == config_name
        ),
        None,
    )
    if recipe is None:
        raise RuntimeError(f"{SHARED_FRONTEND!r} does not serve config {config_name!r}")
    timings: dict[str, float] = {}
    t = time.time()
    url = _frontend_function("server").get_web_url()
    if not url:
        raise RuntimeError(f"deployed {SHARED_FRONTEND!r} frontend has no web url")
    timings["provision"] = time.time() - t
    timings.update(
        _run_rl_step(
            url,
            os.environ["TINKER_API_KEY"],
            recipe["model"],
            recipe["parameterization"],
        )
    )
    return timings, SHARED_FRONTEND, url


def _probe_fft_scoped(engine_name: str) -> tuple[dict[str, float], str, str | None]:
    import spindle
    import spindle.engines

    timings: dict[str, float] = {}
    engine = getattr(spindle.engines, engine_name)()
    t = time.time()
    with spindle.run(engine=engine, name="spindle-synmon-fft") as (url, api_key):
        timings["provision"] = time.time() - t
        timings.update(_run_rl_step(url, api_key, engine.model, "full"))
        t = time.time()
    timings["teardown"] = time.time() - t
    return timings, "spindle-synmon-fft", url


_PROBERS = {
    "lora-shared": _probe_lora_shared,
    "fft-scoped": _probe_fft_scoped,
}


def _run_with_timeout(fn, timeout_s: int):
    import signal

    def _raise(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _targets() -> list[tuple[str, str]]:
    out = [("lora-shared", row["recipe"]["name"]) for row in _served_configs()]
    out += [("fft-scoped", name) for name in FFT_ENGINES]
    return out


@app.function(
    image=probe_image,
    timeout=MONITOR_TIMEOUT_S,
    secrets=[api_secret, hf_secret],
)
def probe(kind: str, target: str) -> dict:
    config = next((c for c in PROBE_KINDS if c.name == kind), None)
    if config is None:
        raise ValueError(f"unknown probe kind {kind!r}")
    name = f"{kind}/{target}"
    print(f"synmon: probe={name!r}")
    try:
        timings, app_name, url = _run_with_timeout(
            lambda: _PROBERS[kind](target), config.timeout_s
        )
        point = RunPoint(
            ts=time.time(),
            timings=timings,
            app_name=app_name,
            total_duration_s=sum(timings.values()),
            status="success",
            url=url,
        )
        return _row(name, point, _record(name, point), None)
    except Exception as exc:
        error = _format_error(exc, name, config.timeout_s)
        print(f"error: probe failed for {name}: {error}")
        point = RunPoint(
            ts=time.time(),
            timings={},
            app_name=name,
            total_duration_s=0.0,
            status="failed",
        )
        return _row(name, point, _record(name, point), error)


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("17 0 * * 0"),
    secrets=[slack_secret],
)
def launch_weekly(name: str = "") -> list[dict]:
    targets = _targets()
    if name:
        targets = [t for t in targets if f"{t[0]}/{t[1]}" == name or t[0] == name]
    rows: list[dict] = []
    payloads = list(
        probe.map(
            [t[0] for t in targets],
            [t[1] for t in targets],
            return_exceptions=True,
        )
    )
    for (kind, target), payload in zip(targets, payloads):
        if isinstance(payload, Exception):
            tb = "".join(
                traceback.format_exception(
                    type(payload), payload, payload.__traceback__
                )
            )
            probe_name = f"{kind}/{target}"
            print(f"error: probe failed for {probe_name}:\n{tb}")
            rows.append(
                {
                    "probe": probe_name,
                    "status": "failed",
                    "total_duration_s": None,
                    "delta": "n/a",
                    "url": None,
                    "timings": {},
                    "error": f"{type(payload).__name__}: {payload}",
                }
            )
            continue
        rows.append(payload)

    _post_report(rows, channel=REPORT_CHANNEL_ID)
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError("synmon fan-out finished with failures")
    return rows


@app.local_entrypoint()
def main(name: str = "") -> None:
    print(launch_weekly.remote(name=name))
