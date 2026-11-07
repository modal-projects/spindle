from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import random
from collections.abc import Awaitable, Callable, Mapping

import modal
from grpclib.exceptions import StreamTerminatedError

from ..contracts import InsertResult

SESSION_KV_NAME = "spindle-sessions"
MODEL_KV_NAME = "spindle-models"
ENGINE_KV_NAME = "spindle-engines"
SAMPLING_SESSION_KV_NAME = "spindle-sampling-sessions"
EXPORT_KV_NAME = "spindle-exports"
ARTIFACT_KV_NAME = "spindle-artifacts"
FFT_POOL_KV_NAME = "spindle-fft-pools"
DEPLOYMENT_KV_NAME = "spindle-deployments"
QUEUE_EVENT_KV_NAME = "spindle-queue-events"
LIST_ITEMS_ATTEMPTS = 3
# A healthy scan of ~4,800 session records can take over 40 seconds.
LIST_ITEMS_TIMEOUT_SECONDS = 120.0
DeleteDict = Callable[..., Awaitable[None]]

STORE_NAMES = {
    "sessions": SESSION_KV_NAME,
    "models": MODEL_KV_NAME,
    "engines": ENGINE_KV_NAME,
    "sampling_sessions": SAMPLING_SESSION_KV_NAME,
    "exports": EXPORT_KV_NAME,
    "artifacts": ARTIFACT_KV_NAME,
    "queue_events": QUEUE_EVENT_KV_NAME,
}

KEY_STORES = {
    "session": "sessions",
    "session_last_seen": "sessions",
    "session_closed": "sessions",
    "model_creation": "models",
    "model": "models",
    "placement": "models",
    "placement_claim": "models",
    "trainer_demand": "models",
    "queue_event": "queue_events",
    "trainer_slot_wait": "models",
    "engine_instance": "engines",
    "model_assets_ready": "engines",
    "model_assets_call": "engines",
    "lora_pool": "engines",
    "sampler_warm": "engines",
    "deployment_failure": "engines",
    "app_deploy": "engines",
    "registry_version": "engines",
    "registry_claim": "engines",
    "engine_call": "engines",
    "engine_current": "engines",
    "trainer_plan": "engines",
    "trainer_reconcile": "engines",
    "trainer_reconcile_request": "engines",
    "trainer_reconcile_complete": "engines",
    "sampling_session_creation": "sampling_sessions",
    "sampling_session": "sampling_sessions",
    "sampler_export_submission": "exports",
    "sampler_export_result": "exports",
    "sampler_artifact": "artifacts",
}


def current_app_id() -> str:
    app = modal.App._get_container_app()
    if app is None or app.app_id is None:
        raise RuntimeError("Modal App ID is unavailable")
    return app.app_id


def app_store_name(name: str, app_id: str) -> str:
    return f"{app_id}-{name.removeprefix('spindle-')}"


class ModalKeyValueStore:
    def __init__(self, dictionary: modal.Dict) -> None:
        self.dictionary = dictionary

    async def get(self, key: str) -> object | None:
        return await self.dictionary.get.aio(key)

    async def put(self, key: str, value: object) -> None:
        await self.dictionary.put.aio(key, value)

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        created = await self.dictionary.put.aio(key, value, skip_if_exists=True)
        if created:
            return InsertResult(True, value)
        return InsertResult(False, await self.dictionary.get.aio(key))

    async def delete(self, key: str) -> None:
        await self.dictionary.pop.aio(key, None)

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key, _ in await self.list_items(prefix))

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]:
        for attempt in range(LIST_ITEMS_ATTEMPTS):
            try:
                async with asyncio.timeout(LIST_ITEMS_TIMEOUT_SECONDS):
                    items = [
                        (key, value)
                        async for key, value in self.dictionary.items.aio()
                        if any(key.startswith(prefix) for prefix in prefixes)
                    ]
                return tuple(sorted(items))
            except (StreamTerminatedError, TimeoutError):
                if attempt == LIST_ITEMS_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(0.2 * 2**attempt)
        raise AssertionError("unreachable")


def session_task_store_name(session_id: str, app_id: str) -> str:
    if not session_id:
        raise ValueError("session_id must be non-empty")
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:24]
    return f"{app_id}-tasks-{digest}"


class ModalSessionKeyValueStores:
    def __init__(
        self,
        app_id: str | None = None,
        delete: DeleteDict | None = None,
    ) -> None:
        self.app_id = app_id or current_app_id()
        self.stores: dict[str, ModalKeyValueStore] = {}
        self.delete = delete or modal.Dict.objects.delete.aio

    def for_session(self, session_id: str) -> ModalKeyValueStore:
        store = self.stores.get(session_id)
        if store is None:
            store = ModalKeyValueStore(
                modal.Dict.from_name(
                    session_task_store_name(session_id, self.app_id),
                    create_if_missing=True,
                )
            )
            self.stores[session_id] = store
        return store

    async def delete_session(self, session_id: str) -> None:
        self.stores.pop(session_id, None)
        await self.delete(
            session_task_store_name(session_id, self.app_id),
            allow_missing=True,
        )


class RoutedKeyValueStore:
    def __init__(self, stores: dict[str, ModalKeyValueStore]) -> None:
        self.stores = stores

    def _store(self, key: str) -> ModalKeyValueStore:
        family = key.partition(":")[0]
        try:
            return self.stores[KEY_STORES[family]]
        except KeyError:
            raise ValueError(f"unknown key family: {family}") from None

    def _matching_stores(
        self, prefixes: tuple[str, ...]
    ) -> tuple[ModalKeyValueStore, ...]:
        domains = {
            domain
            for family, domain in KEY_STORES.items()
            if any(
                f"{family}:".startswith(prefix) or prefix.startswith(f"{family}:")
                for prefix in prefixes
            )
        }
        if not domains:
            raise ValueError(f"unknown key prefixes: {prefixes}")
        return tuple(self.stores[domain] for domain in sorted(domains))

    async def get(self, key: str) -> object | None:
        return await self._store(key).get(key)

    async def put(self, key: str, value: object) -> None:
        await self._store(key).put(key, value)

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        return await self._store(key).put_if_absent(key, value)

    async def delete(self, key: str) -> None:
        await self._store(key).delete(key)

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key, _ in await self.list_items(prefix))

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]:
        items = []
        for store in self._matching_stores(prefixes):
            items.extend(await store.list_items(*prefixes))
        return tuple(sorted(items))


def shared_kv(
    app_id: str | None = None, environment: str | None = None
) -> RoutedKeyValueStore:
    app_id = app_id or os.environ.get("SPINDLE_FRONTEND_APP_ID") or current_app_id()
    return RoutedKeyValueStore(
        {
            domain: ModalKeyValueStore(
                modal.Dict.from_name(
                    app_store_name(name, app_id),
                    environment_name=environment,
                    create_if_missing=True,
                )
            )
            for domain, name in STORE_NAMES.items()
        }
    )


def fft_pool_kv(app_id: str | None = None) -> ModalKeyValueStore:
    app_id = app_id or current_app_id()
    return ModalKeyValueStore(
        modal.Dict.from_name(
            app_store_name(FFT_POOL_KV_NAME, app_id),
            create_if_missing=True,
        )
    )


VOLUME_RETRY_DELAYS_S = (0.5, 1.0, 2.0, 4.0)


async def _retry_rate_limited(call: Callable[[], Awaitable[object]]) -> object:
    for delay in (*VOLUME_RETRY_DELAYS_S, None):
        try:
            return await call()
        except modal.exception.ResourceExhaustedError:
            if delay is None:
                raise
            await asyncio.sleep(delay * (1 + random.random()))


class ModalVolumeStore:
    """One `<folder>/<key>.json` file per key in a shared Modal Volume."""

    def __init__(self, volume: modal.Volume, folder: str) -> None:
        self.volume = volume
        self.folder = folder

    def _path(self, key: str) -> str:
        return f"/{self.folder}/{key}.json"

    def read(self, key: str) -> object | None:
        try:
            return json.loads(b"".join(self.volume.read_file(self._path(key))))
        except FileNotFoundError:
            return None

    async def get(self, key: str) -> object | None:
        return await _retry_rate_limited(lambda: asyncio.to_thread(self.read, key))

    async def put(self, key: str, value: object) -> None:
        await self.put_many({key: value})

    async def put_many(self, items: Mapping[str, object]) -> None:
        """Write every item in one Volume commit."""
        if not items:
            return
        async with self.volume.batch_upload.aio(force=True) as batch:
            for key, value in items.items():
                batch.put_file(io.BytesIO(json.dumps(value).encode()), self._path(key))

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        current = await self.get(key)
        if current is not None:
            return InsertResult(False, current)
        await self.put(key, value)
        return InsertResult(True, value)

    async def delete(self, key: str) -> None:
        try:
            await self.volume.remove_file.aio(self._path(key))
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass

    async def clear(self) -> None:
        try:
            await self.volume.remove_file.aio(f"/{self.folder}", recursive=True)
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        try:
            entries = await _retry_rate_limited(
                lambda: self.volume.listdir.aio(f"/{self.folder}")
            )
        except (FileNotFoundError, modal.exception.NotFoundError):
            return ()
        names = [os.path.basename(entry.path) for entry in entries]
        return tuple(
            sorted(
                name.removesuffix(".json")
                for name in names
                if name.endswith(".json") and name.startswith(prefix)
            )
        )

    async def list_items(self, *prefixes: str) -> tuple[tuple[str, object], ...]:
        keys = [
            key
            for key in await self.list_keys("")
            if any(key.startswith(prefix) for prefix in prefixes)
        ]
        values = await asyncio.gather(*(self.get(key) for key in keys))
        return tuple(zip(keys, values))


def deployment_kv(
    app_id: str | None = None, environment: str | None = None
) -> ModalVolumeStore:
    """The recipes a frontend serves: `<app_id>/<recipe>.json` in one shared Volume.

    Keyed by app ID, so a frontend redeployed after `spindle stop` starts empty.
    """
    return ModalVolumeStore(
        modal.Volume.from_name(
            DEPLOYMENT_KV_NAME, environment_name=environment, create_if_missing=True
        ),
        app_id or current_app_id(),
    )
