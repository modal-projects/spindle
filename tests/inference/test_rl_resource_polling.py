import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from scripts.benchmarks.lora_routing import supervise
from scripts.benchmarks.rl_rust import run


def test_resource_polling_keeps_history_without_refetching_it(monkeypatch):
    requested = []
    active = peak = 0

    async def info(request):
        nonlocal active, peak
        requested.append(request.task_id)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return SimpleNamespace(
            info=SimpleNamespace(
                started_at=10,
                finished_at=0,
                gpu_type="H200",
                gpu_config=SimpleNamespace(count=1),
            )
        )

    tasks = [
        SimpleNamespace(task_id=name, app_description="ours")
        for name in ("finished", "active", "new")
    ]
    stub = SimpleNamespace(
        TaskList=AsyncMock(return_value=SimpleNamespace(tasks=tasks)),
        TaskGetInfo=info,
    )
    monkeypatch.setattr(
        supervise,
        "_Client",
        SimpleNamespace(from_env=AsyncMock(return_value=SimpleNamespace(stub=stub))),
    )
    historical = dict(finished_at=9, app="ours", gpu_count=8)
    known = dict(finished=historical, active=dict(finished_at=0, app="ours"))
    result = asyncio.run(supervise.resources.aio("test", ["ours"], known))
    assert set(requested) == {"active", "new"}
    assert peak == 2
    assert result["finished"] == historical
    assert result["new"]["app"] == "ours"


def test_monitor_timeouts_preserve_run_and_retry(monkeypatch, tmp_path):
    state = dict(started=run.time.time(), phase="training", updates={"default": 2})
    known = {"trainer": dict(finished_at=0, gpu_count=8)}
    reads = AsyncMock(
        side_effect=[TimeoutError(), TimeoutError(), TimeoutError(), known]
    )
    monkeypatch.setattr(run, "resources", SimpleNamespace(aio=reads))

    async def pause(seconds):
        saved = json.loads((tmp_path / "status.json").read_text())
        assert saved["updates"] == {"default": 2}
        assert saved["resource_poll_error"].startswith("TimeoutError")
        assert not (tmp_path / "STOP").exists()

    monkeypatch.setattr(run.asyncio, "sleep", pause)
    assert asyncio.run(run.observe_resources(tmp_path, ["ours"], known, state)) == known
    assert reads.await_count == 4
    assert "resource_poll_error" not in state


def test_monitor_outage_has_a_deadline(monkeypatch, tmp_path):
    state = dict(started=run.time.time())
    clock = [0.0]
    reads = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(run, "resources", SimpleNamespace(aio=reads))
    monkeypatch.setattr(run.time, "monotonic", lambda: clock[0])

    async def pause(seconds):
        clock[0] += 300

    monkeypatch.setattr(run.asyncio, "sleep", pause)
    with pytest.raises(RuntimeError, match="ten minutes"):
        asyncio.run(run.observe_resources(tmp_path, ["ours"], {}, state))
