import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from spindle.control_plane.records import SessionRecord
from spindle.providers.local.kv import InMemoryKeyValueStore
from spindle.providers.modal import kv


@pytest.mark.parametrize(
    "explicit,expected", [(None, "ap-frontend"), ("ap-explicit", "ap-explicit")]
)
def test_independent_trainer_uses_frontend_registry(monkeypatch, explicit, expected):
    names = []
    monkeypatch.setenv("SPINDLE_FRONTEND_APP_ID", "ap-frontend")
    monkeypatch.setattr(kv, "current_app_id", lambda: "ap-trainer")
    monkeypatch.setattr(
        kv.modal.Dict, "from_name", lambda name, **kwargs: names.append(name)
    )
    kv.shared_kv(explicit)
    assert set(names) == {
        kv.app_store_name(name, expected) for name in kv.STORE_NAMES.values()
    }


class RetryableStreamError(Exception):
    pass


def test_modal_store_validates_typed_records() -> None:
    async def run() -> None:
        values = {}

        async def get(key):
            return values.get(key)

        async def put(key, value, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

        store = kv.ModalKeyValueStore(
            SimpleNamespace(
                get=SimpleNamespace(aio=get),
                put=SimpleNamespace(aio=put),
            )
        )
        session = SessionRecord(session_id="session", created_at=1.0)
        await store.put("session:session", session)

        assert values["session:session"] == session.model_dump(mode="json")
        assert await store.get("session:session") == session

    asyncio.run(run())


class StreamingItems:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0

    def aio(self):
        async def stream():
            self.attempts += 1
            if self.attempts <= self.failures:
                raise RetryableStreamError
            yield "placement:model-a", {"instance": "engine-a"}
            yield "model:model-a", {"state": "ready"}

        return stream()


class PartiallyTerminatedItems:
    def __init__(self) -> None:
        self.attempts = 0

    def aio(self):
        async def stream():
            self.attempts += 1
            yield "placement:model-a", {"instance": "engine-a"}
            if self.attempts == 1:
                raise RetryableStreamError
            yield "placement:model-b", {"instance": "engine-b"}

        return stream()


class HangingItems:
    def __init__(self) -> None:
        self.attempts = 0

    def aio(self):
        async def stream():
            self.attempts += 1
            await asyncio.Event().wait()
            yield "unreachable", {}

        return stream()


def test_list_items_retries_terminated_stream(monkeypatch) -> None:
    async def run() -> None:
        items = StreamingItems(failures=2)
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items), record_types={})
        sleep = AsyncMock()
        monkeypatch.setattr(kv, "StreamTerminatedError", RetryableStreamError)
        monkeypatch.setattr(kv.asyncio, "sleep", sleep)

        assert await store.list_items("placement:") == (
            ("placement:model-a", {"instance": "engine-a"}),
        )
        assert items.attempts == 3
        assert [call.args[0] for call in sleep.await_args_list] == [0.2, 0.4]

    asyncio.run(run())


def test_list_items_discards_partial_retry(monkeypatch) -> None:
    async def run() -> None:
        items = PartiallyTerminatedItems()
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items), record_types={})
        monkeypatch.setattr(kv, "StreamTerminatedError", RetryableStreamError)
        monkeypatch.setattr(kv.asyncio, "sleep", AsyncMock())

        assert await store.list_items("placement:") == (
            ("placement:model-a", {"instance": "engine-a"}),
            ("placement:model-b", {"instance": "engine-b"}),
        )
        assert items.attempts == 2

    asyncio.run(run())


def test_list_items_retries_stalled_stream(monkeypatch) -> None:
    async def run() -> None:
        items = HangingItems()
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items), record_types={})
        monkeypatch.setattr(kv, "LIST_ITEMS_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(kv.asyncio, "sleep", AsyncMock())

        with pytest.raises(TimeoutError):
            await store.list_items("engine_instance:")
        assert items.attempts == 3

    asyncio.run(run())


def test_list_items_does_not_retry_other_errors(monkeypatch) -> None:
    async def run() -> None:
        items = StreamingItems(failures=1)
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items), record_types={})
        monkeypatch.setattr(kv, "StreamTerminatedError", ValueError)
        sleep = AsyncMock()
        monkeypatch.setattr(kv.asyncio, "sleep", sleep)

        with pytest.raises(RetryableStreamError):
            await store.list_items("placement:")
        assert items.attempts == 1
        sleep.assert_not_awaited()

    asyncio.run(run())


def test_list_items_reraises_after_retry_limit(monkeypatch) -> None:
    async def run() -> None:
        items = StreamingItems(failures=3)
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items), record_types={})
        monkeypatch.setattr(kv, "StreamTerminatedError", RetryableStreamError)
        monkeypatch.setattr(kv.asyncio, "sleep", AsyncMock())

        with pytest.raises(RetryableStreamError):
            await store.list_items("engine_instance:")
        assert items.attempts == 3

    asyncio.run(run())


def test_lora_pool_registry_supports_publication_and_cleanup():
    async def run():
        stores = {name: InMemoryKeyValueStore() for name in kv.STORE_NAMES}
        routed = kv.RoutedKeyValueStore(stores)
        key = "lora_pool:spindle-test"
        value = {"definition_id": "test", "touched_at": 1.0}
        await routed.put(key, value)
        assert await routed.get(key) == value
        assert await routed.list_items("lora_pool:") == ((key, value),)
        await routed.delete(key)
        assert await routed.list_items("lora_pool:") == ()

    asyncio.run(run())
