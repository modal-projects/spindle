import asyncio
from types import SimpleNamespace

from spindle.providers.modal.capacity import CHECK_INTERVAL_S, gpu_label, gpu_wait


class FakeFunction:
    def __init__(self, backlog: int, runners: int) -> None:
        self.backlog = backlog
        self.runners = runners
        self.calls = 0
        self.get_current_stats = SimpleNamespace(aio=self._stats)

    async def _stats(self) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(backlog=self.backlog, num_total_runners=self.runners)


def test_gpu_label_lists_fallbacks_and_nodes() -> None:
    assert gpu_label("H200", 8) == "8× H200"
    assert gpu_label(["H200", "B300", "B200"], 8) == "8× H200 or B300 or B200"
    assert gpu_label("H200", 8, nodes=2) == "2 nodes of 8× H200"


def test_gpu_wait_reports_only_queued_work_without_containers() -> None:
    async def run() -> None:
        function = FakeFunction(backlog=1, runners=0)
        now = [0.0]
        wait = gpu_wait(lambda _: function, lambda _: "8× H200", clock=lambda: now[0])

        assert await wait("qwen") == "8× H200"

        function.runners = 1
        assert await wait("qwen") == "8× H200"
        assert function.calls == 1

        now[0] += CHECK_INTERVAL_S
        assert await wait("qwen") is None

        function.backlog, function.runners = 0, 0
        now[0] += CHECK_INTERVAL_S
        assert await wait("qwen") is None
        assert function.calls == 3

    asyncio.run(run())
