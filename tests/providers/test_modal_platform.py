import asyncio
from unittest.mock import AsyncMock, patch

import modal
import pytest

from spindle.providers.local import InMemoryKeyValueStore
from spindle.providers.modal.engines import (
    EngineInstanceRecord,
    ModalEnginePlatform,
    call_key,
    current_instance_key,
    instance_key,
)

DEFINITION = "qwen3_test"


def recording_platform() -> tuple[
    ModalEnginePlatform,
    InMemoryKeyValueStore,
    list[str],
]:
    kv = InMemoryKeyValueStore()
    spawned: list[str] = []

    async def spawn(definition_id: str, instance_id: str) -> str:
        spawned.append(instance_id)
        return f"fc-{instance_id}"

    return ModalEnginePlatform(kv, spawn), kv, spawned


def test_spawn_instance_adds_replicas() -> None:
    async def run() -> None:
        platform, kv, spawned = recording_platform()
        first = await platform.spawn_instance(DEFINITION)
        second = await platform.spawn_instance(DEFINITION)
        assert spawned == [first.instance_id, second.instance_id]
        assert first.instance_id != second.instance_id
        assert await kv.get(current_instance_key(DEFINITION)) is None
        assert await kv.get(instance_key(first.instance_id)) is not None
        assert await kv.get(instance_key(second.instance_id)) is not None

    asyncio.run(run())


def test_first_ensure_spawns_and_commits_call_record() -> None:
    async def run() -> None:
        platform, kv, spawned = recording_platform()
        instance = await platform.ensure_instance(DEFINITION)
        assert spawned == [instance.instance_id]
        assert instance.state == "starting"
        assert await kv.get(current_instance_key(DEFINITION)) == instance.instance_id
        assert (
            await kv.get(call_key(instance.instance_id)) == f"fc-{instance.instance_id}"
        )

    asyncio.run(run())


def test_claimed_but_never_spawned_instance_is_replaced() -> None:
    async def run() -> None:
        platform, kv, spawned = recording_platform()
        await kv.put(current_instance_key(DEFINITION), "engine-ghost")
        instance = await platform.ensure_instance(DEFINITION)
        assert instance.instance_id != "engine-ghost"
        assert spawned == [instance.instance_id]
        assert await kv.get(current_instance_key(DEFINITION)) == instance.instance_id

    asyncio.run(run())


async def probed_state(kv: InMemoryKeyValueStore, poll_error: Exception) -> str:
    platform = ModalEnginePlatform(kv, AsyncMock())
    record = EngineInstanceRecord(
        instance_id="engine-a",
        definition_id=DEFINITION,
        revision="img-1",
        state="running",
        call_id="fc-engine-a",
    )
    await kv.put(instance_key("engine-a"), record.model_dump(mode="json"))
    call = AsyncMock()
    call.get.aio.side_effect = poll_error
    with patch.object(modal.FunctionCall, "from_id", return_value=call):
        return (await platform.get_instance("engine-a")).state


def test_stop_instance_cancels_via_record_call_id_fallback() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        platform = ModalEnginePlatform(kv, AsyncMock())
        record = EngineInstanceRecord(
            instance_id="engine-a",
            definition_id=DEFINITION,
            revision="img-1",
            state="running",
            call_id="fc-engine-a",
        )
        await kv.put(instance_key("engine-a"), record.model_dump(mode="json"))
        call = AsyncMock()
        with patch.object(modal.FunctionCall, "from_id", return_value=call) as from_id:
            await platform.stop_instance("engine-a")
        from_id.assert_called_once_with("fc-engine-a")
        call.cancel.aio.assert_awaited_once_with(terminate_containers=True)
        assert await kv.get(instance_key("engine-a")) is None

    asyncio.run(run())


def test_client_is_rebuilt_when_instance_reregisters() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        platform = ModalEnginePlatform(kv, AsyncMock())
        record = EngineInstanceRecord(
            instance_id="engine-a",
            definition_id=DEFINITION,
            revision="img-1",
            state="running",
            call_id="fc-engine-a",
            url="http://old-tunnel",
            token="old",
        )
        await kv.put(instance_key("engine-a"), record.model_dump(mode="json"))
        call = AsyncMock()
        call.get.aio.side_effect = TimeoutError()
        with patch.object(modal.FunctionCall, "from_id", return_value=call):
            assert (await platform.get_instance("engine-a")).boot_id == ""
            first = platform.client("engine-a")
            assert platform.client("engine-a") is first
            restarted = record.model_copy(
                update={"url": "http://new-tunnel", "token": "new", "boot_id": "b2"}
            )
            await kv.put(instance_key("engine-a"), restarted.model_dump(mode="json"))
            assert (await platform.get_instance("engine-a")).boot_id == "b2"
            second = platform.client("engine-a")
        assert second is not first
        assert str(second.http.base_url) == "http://new-tunnel"
        assert second.http.headers["authorization"] == "Bearer new"

    asyncio.run(run())


def test_transient_poll_errors_are_not_death_verdicts() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        state = await probed_state(kv, modal.exception.ConnectionError("blip"))
        assert state == "running"
        state = await probed_state(kv, RuntimeError("call outcome"))
        assert state == "dead"

    asyncio.run(run())


def test_concurrent_instance_reads_share_only_pending_liveness_check():
    async def run():
        platform, kv, _ = recording_platform()
        record = EngineInstanceRecord(
            instance_id="shared",
            definition_id=DEFINITION,
            revision="one",
            state="running",
            call_id="fc-shared",
        )
        await kv.put(instance_key("shared"), record.model_dump(mode="json"))
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def poll(*args, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            raise TimeoutError

        call = AsyncMock()
        call.get.aio.side_effect = poll
        with patch.object(modal.FunctionCall, "from_id", return_value=call):
            first = asyncio.create_task(platform.get_instance("shared"))
            await entered.wait()
            second = asyncio.create_task(platform.get_instance("shared"))
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            assert (await second).state == "running"
            assert calls == 1
            assert not platform._instance_reads
            # No cache survives completion; the next read must see a restart.
            restarted = record.model_copy(update={"boot_id": "new-boot"})
            await kv.put(instance_key("shared"), restarted.model_dump(mode="json"))
            assert (await platform.get_instance("shared")).boot_id == "new-boot"
            assert calls == 2

    asyncio.run(run())
