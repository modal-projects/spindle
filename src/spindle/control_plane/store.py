from __future__ import annotations

from pydantic import BaseModel

from spindle.providers.contracts import InsertResult, KeyValueStore

from .records import (
    ModelCreationRecord,
    ModelRecord,
    PlacementRecord,
    SamplerArtifactRecord,
    SamplerExportResultRecord,
    SamplerExportSubmissionRecord,
    SampleTaskRecord,
    SamplingSessionCreationRecord,
    SamplingSessionRecord,
    SessionClosedRecord,
    SessionLastSeenRecord,
    SessionRecord,
)

RECORD_TYPES: dict[str, type[BaseModel]] = {
    "session": SessionRecord,
    "session_last_seen": SessionLastSeenRecord,
    "session_closed": SessionClosedRecord,
    "model_creation": ModelCreationRecord,
    "model": ModelRecord,
    "placement": PlacementRecord,
    "sampling_session_creation": SamplingSessionCreationRecord,
    "sampling_session": SamplingSessionRecord,
    "sampler_export_submission": SamplerExportSubmissionRecord,
    "sampler_export_result": SamplerExportResultRecord,
    "sampler_artifact": SamplerArtifactRecord,
    "sample_task": SampleTaskRecord,
}


class TypedKeyValueStore:
    """Validate durable control-plane records at the KV boundary."""

    def __init__(self, store: KeyValueStore) -> None:
        self.store = store

    def _record_type(self, key: str) -> type[BaseModel] | None:
        return RECORD_TYPES.get(key.partition(":")[0])

    def _decode(self, key: str, value: object) -> object:
        record_type = self._record_type(key)
        return record_type.model_validate(value) if record_type is not None else value

    def _encode(self, key: str, value: object) -> object:
        record_type = self._record_type(key)
        if record_type is None:
            return (
                value.model_dump(mode="json") if isinstance(value, BaseModel) else value
            )
        return record_type.model_validate(value).model_dump(mode="json")

    async def get(self, key: str) -> object | None:
        value = await self.store.get(key)
        return None if value is None else self._decode(key, value)

    async def put(self, key: str, value: object) -> None:
        await self.store.put(key, self._encode(key, value))

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        result = await self.store.put_if_absent(key, self._encode(key, value))
        return InsertResult(result.created, self._decode(key, result.value))

    async def delete(self, key: str) -> None:
        await self.store.delete(key)

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        return await self.store.list_keys(prefix)

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]:
        return tuple(
            (key, self._decode(key, value))
            for key, value in await self.store.list_items(*prefixes)
        )


def typed_kv(store: KeyValueStore) -> TypedKeyValueStore:
    return store if isinstance(store, TypedKeyValueStore) else TypedKeyValueStore(store)
