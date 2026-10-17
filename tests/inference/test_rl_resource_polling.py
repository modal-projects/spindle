import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from scripts.benchmarks.lora_routing import supervise


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
