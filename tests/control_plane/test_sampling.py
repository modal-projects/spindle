import asyncio
import time

import pytest

from tests.support import EchoExecutor, TinkerStubSampler
from spindle.control_plane import ControlPlane, FutureResolutionStatus
from spindle.control_plane.service import SAMPLER_WARM_FOR_S
from spindle.control_plane.keys import (
    sample_task_key,
    sampling_session_creation_key,
    sampling_session_key,
    session_key,
    session_last_seen_key,
)
from spindle.errors import RecordNotFound, RecordUnavailable, SequenceConflict
from spindle.providers import SamplingTask
from spindle.providers.local import (
    InMemoryKeyValueStore,
    InMemorySessionKeyValueStores,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)

DEFINITION = "qwen3_8b"
BASE_MODEL = "Qwen/Qwen3-8B"


async def sampling_plane(runner=None):
    if runner is None:
        runner = TinkerStubSampler()
    tasks = LocalSamplingTaskPlatform(runner)
    task_stores = InMemorySessionKeyValueStores()
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor),
        sampling_tasks=tasks,
        sampling_task_stores=task_stores,
    )
    session = await plane.create_session()
    sampling = await plane.create_sampling_session(
        session_id=session.session_id,
        sampling_session_seq_id=0,
        base_model=BASE_MODEL,
        engine_definition_id=DEFINITION,
    )
    return plane, tasks, session.session_id, sampling.sampling_session_id


def sample_request(sampling_session_id: str, seq_id: int = 0) -> dict:
    return {
        "type": "sample",
        "sampling_session_id": sampling_session_id,
        "seq_id": seq_id,
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        "sampling_params": {"max_tokens": 2},
    }


def test_sampling_sessions_and_submissions_are_idempotent() -> None:
    async def run() -> None:
        plane, tasks, session_id, sampling_session_id = await sampling_plane()
        duplicate = await plane.create_sampling_session(
            session_id=session_id,
            sampling_session_seq_id=0,
            base_model=BASE_MODEL,
        )
        assert duplicate.sampling_session_id == sampling_session_id

        request = sample_request(sampling_session_id)
        request_id = await plane.submit_sample(request)
        assert await plane.submit_sample(request) == request_id
        resolution = await plane.retrieve(request_id, timeout=1.0)
        assert resolution.status == FutureResolutionStatus.COMPLETE
        assert resolution.result["sequences"][0]["tokens"] == [0, 1]
        assert len(tasks._tasks) == 1

        changed = sample_request(sampling_session_id)
        changed["num_samples"] = 2
        with pytest.raises(SequenceConflict):
            await plane.submit_sample(changed)

    asyncio.run(run())


def test_cache_affinity_key_participates_in_sample_idempotency() -> None:
    async def run() -> None:
        plane, _, _, sampling_session_id = await sampling_plane()
        request = sample_request(sampling_session_id)
        request["cache_affinity_key"] = "trajectory-a"
        request_id = await plane.submit_sample(request)

        assert await plane.submit_sample(request) == request_id
        conflicting = {**request, "cache_affinity_key": "trajectory-b"}
        with pytest.raises(SequenceConflict):
            await plane.submit_sample(conflicting)

    asyncio.run(run())


def test_session_collection_removes_task_store() -> None:
    async def run() -> None:
        plane, _, session_id, sampling_session_id = await sampling_plane()
        request_id = await plane.submit_sample(sample_request(sampling_session_id))
        task_stores = plane.sampling_task_stores
        assert isinstance(task_stores, InMemorySessionKeyValueStores)
        store = task_stores.for_session(session_id)
        assert await store.get(sample_task_key(sampling_session_id, 0)) is not None

        await plane.close_session(session_id, "done")
        with pytest.raises(RecordUnavailable):
            await plane.get_sampling_session(sampling_session_id)
        with pytest.raises(RecordUnavailable):
            await plane.retrieve(request_id)
        await plane.sweep_idle_sessions(0)

        assert session_id not in task_stores.stores
        assert await plane.kv.get(sampling_session_key(sampling_session_id)) is not None
        assert (
            await plane.kv.get(sampling_session_creation_key(session_id, 0)) is not None
        )
        with pytest.raises(RecordNotFound):
            await plane.get_sampling_session(sampling_session_id)
        with pytest.raises(RecordNotFound):
            await plane.submit_sample(sample_request(sampling_session_id))
        with pytest.raises(RecordNotFound):
            await plane.retrieve(request_id)
        assert session_id not in task_stores.stores

    asyncio.run(run())


@pytest.mark.parametrize("session_task_stores", [True, False])
def test_session_sweep_never_lists_sampling_metadata(session_task_stores) -> None:
    class LifecycleStore(InMemoryKeyValueStore):
        async def list_items(self, *prefixes):
            assert set(prefixes) <= {
                "session:",
                "session_last_seen:",
                "session_closed:",
                "model:",
                "model_creation:",
                "placement:",
            }
            return await super().list_items(*prefixes)

    async def run() -> None:
        kv = LifecycleStore()
        plane = ControlPlane(
            kv,
            LocalEnginePlatform(DEFINITION, EchoExecutor),
            sampling_task_stores=(
                InMemorySessionKeyValueStores() if session_task_stores else None
            ),
        )
        session = await plane.create_session()
        for i in range(5000):
            await kv.put(f"sampling_session:{i}", {"unrelated": "metadata"})
        assert await plane.sweep_idle_sessions(0) == (session.session_id,)
        assert await kv.get(session_key(session.session_id)) is None
        assert await kv.get("sampling_session:0") is not None

    asyncio.run(run())


def test_task_store_delete_failure_retains_closed_session() -> None:
    class FailingStores(InMemorySessionKeyValueStores):
        async def delete_session(self, session_id: str) -> None:
            raise RuntimeError(session_id)

    async def run() -> None:
        kv = InMemoryKeyValueStore()
        stores = FailingStores()
        plane = ControlPlane(
            kv,
            LocalEnginePlatform(DEFINITION, EchoExecutor),
            sampling_tasks=LocalSamplingTaskPlatform(TinkerStubSampler()),
            sampling_task_stores=stores,
        )
        session = await plane.create_session()
        await plane.close_session(session.session_id, "done")

        with pytest.raises(RuntimeError, match=session.session_id):
            await plane.sweep_idle_sessions(0)

        assert await kv.get(session_key(session.session_id)) is not None

    asyncio.run(run())


def test_sampling_task_carries_engine_definition() -> None:
    async def run() -> None:
        seen = []

        async def capture(task: SamplingTask) -> object:
            seen.append(task)
            return {"type": "sample", "sequences": []}

        plane, _, _, sampling_session_id = await sampling_plane(capture)
        request = sample_request(sampling_session_id)
        request["cache_affinity_key"] = "trajectory-a"
        request_id = await plane.submit_sample(request)
        await plane.retrieve(request_id, timeout=1.0)
        assert seen[0].engine_definition_id == DEFINITION
        assert seen[0].session_id
        assert seen[0].payload["cache_affinity_key"] == "trajectory-a"

    asyncio.run(run())


def test_sampling_session_creation_detects_conflicts() -> None:
    async def run() -> None:
        plane, _, session_id, _ = await sampling_plane()
        with pytest.raises(SequenceConflict):
            await plane.create_sampling_session(
                session_id=session_id,
                sampling_session_seq_id=0,
                base_model="Qwen/Qwen3-4B",
            )

    asyncio.run(run())


def test_sample_future_pends_and_lost_tasks_are_retryable() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked(task: SamplingTask) -> object:
            started.set()
            await release.wait()
            return {"type": "sample", "sequences": []}

        plane, tasks, _, sampling_session_id = await sampling_plane(blocked)
        request_id = await plane.submit_sample(sample_request(sampling_session_id))
        await started.wait()
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.PENDING

        task_id = next(iter(tasks._tasks))
        await tasks.forget(task_id)
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.RETRYABLE

    asyncio.run(run())


class CountingStore(InMemoryKeyValueStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []

    async def get(self, key: str):
        self.reads.append(key)
        return await super().get(key)


def test_submit_sample_checks_the_session_once_and_records_the_touch() -> None:
    async def run() -> None:
        kv = CountingStore()
        plane = ControlPlane(
            kv,
            LocalEnginePlatform(DEFINITION, EchoExecutor),
            sampling_tasks=LocalSamplingTaskPlatform(TinkerStubSampler()),
            sampling_task_stores=InMemorySessionKeyValueStores(),
        )
        session = await plane.create_session()
        sampling = await plane.create_sampling_session(
            session_id=session.session_id,
            sampling_session_seq_id=0,
            base_model=BASE_MODEL,
            engine_definition_id=DEFINITION,
        )
        kv.reads.clear()
        await plane.submit_sample(sample_request(sampling.sampling_session_id))
        assert kv.reads.count(session_key(session.session_id)) == 1
        assert await kv.get(session_last_seen_key(session.session_id)) is not None

        await plane.close_session(session.session_id, "done")
        with pytest.raises(RecordUnavailable):
            await plane.submit_sample(sample_request(sampling.sampling_session_id, 1))

    asyncio.run(run())


def test_slow_samples_explain_startup_until_the_pool_completes_one() -> None:
    async def run() -> None:
        gates: dict[tuple[str, int], asyncio.Event] = {}

        async def gated(task: SamplingTask) -> object:
            key = (task.sampling_session_id, task.payload["seq_id"])
            await gates.setdefault(key, asyncio.Event()).wait()
            return {"type": "sample", "sequences": []}

        plane, _, session_id, sampling_session_id = await sampling_plane(gated)
        now = [time.time()]
        plane.clock = lambda: now[0]

        async def slow_sample(sampler: str, seq_id: int = 0):
            request_id = await plane.submit_sample(sample_request(sampler, seq_id))
            now[0] += 45
            return request_id, await plane.retrieve(request_id)

        first, cold = await slow_sample(sampling_session_id)
        assert cold.status == FutureResolutionStatus.PENDING
        assert cold.reason.startswith(
            f"starting a sampler for {BASE_MODEL} (45s so far)"
        )
        gates[(sampling_session_id, 0)].set()
        resolution = await plane.retrieve(first, timeout=5)
        assert resolution.status == FutureResolutionStatus.COMPLETE

        # RL loops open a new sampling session every step on the same warm pool.
        next_step = await plane.create_sampling_session(
            session_id=session_id,
            sampling_session_seq_id=1,
            base_model=BASE_MODEL,
            engine_definition_id=DEFINITION,
        )
        _, warm = await slow_sample(next_step.sampling_session_id)
        assert warm.status == FutureResolutionStatus.PENDING
        assert warm.reason is None

        now[0] += SAMPLER_WARM_FOR_S
        _, idle = await slow_sample(sampling_session_id, 1)
        assert idle.reason.startswith(f"starting a sampler for {BASE_MODEL}")
        for gate in gates.values():
            gate.set()

    asyncio.run(run())


def test_base_model_samples_report_the_weight_download() -> None:
    async def run() -> None:
        released = asyncio.Event()

        async def blocked(task: SamplingTask) -> object:
            await released.wait()
            return {"type": "sample", "sequences": []}

        async def downloading(definition_id: str) -> float | None:
            assert definition_id == DEFINITION
            return plane.clock() - 5

        plane, _, _, sampling_session_id = await sampling_plane(blocked)
        plane.downloading_weights = downloading
        request_id = await plane.submit_sample(sample_request(sampling_session_id))
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.PENDING
        assert resolution.reason.startswith(
            f"downloading weights for {BASE_MODEL} (5s so far)"
        )
        released.set()

    asyncio.run(run())
