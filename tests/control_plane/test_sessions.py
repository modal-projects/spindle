import asyncio

import pytest

from spindle.control_plane import ControlPlane
from spindle.control_plane.keys import session_key
from spindle.control_plane.records import SessionRecord
from spindle.errors import RecordNotFound, RecordUnavailable
from spindle.resource_scope import ResourcePolicy, SessionAttribution
from tests.support import EchoExecutor
from spindle.providers.local import (
    LocalEnginePlatform,
    InMemoryKeyValueStore,
)

DEFINITION = "qwen3_4b_lora32_16k"


def control_plane(**kwargs) -> ControlPlane:
    return ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor),
        **kwargs,
    )


def test_session_lifecycle() -> None:
    async def run() -> None:
        times = iter((1.0, 2.0, 3.0))
        plane = control_plane(
            session_id_factory=lambda: "session-1",
            clock=lambda: next(times),
        )
        session = await plane.create_session()
        assert session.created_at == 1.0
        assert (await plane.heartbeat("session-1")).seen_at == 2.0

        closed = await plane.close_session("session-1", "client requested")
        assert closed.closed_at == 3.0
        with pytest.raises(RecordUnavailable):
            await plane.heartbeat("session-1")

    asyncio.run(run())


def test_close_is_idempotent() -> None:
    async def run() -> None:
        plane = control_plane(session_id_factory=lambda: "session-1")
        await plane.create_session()
        first = await plane.close_session("session-1", "first")
        second = await plane.close_session("session-1", "second")
        assert second == first

    asyncio.run(run())


def test_sweep_closes_idle_sessions_only() -> None:
    async def run() -> None:
        now = 0.0
        sessions = iter(("session-idle", "session-live"))
        plane = control_plane(
            session_id_factory=lambda: next(sessions),
            clock=lambda: now,
        )
        await plane.create_session()
        now = 100.0
        await plane.create_session()

        now = 150.0
        assert await plane.sweep_idle_sessions(idle_timeout=60.0) == ("session-idle",)
        with pytest.raises(RecordNotFound):
            await plane.heartbeat("session-idle")
        assert await plane.kv.list_keys("session:") == ("session:session-live",)
        await plane.heartbeat("session-live")
        assert await plane.sweep_idle_sessions(idle_timeout=60.0) == ()

    asyncio.run(run())


def test_unknown_session_is_rejected() -> None:
    async def run() -> None:
        plane = control_plane()
        with pytest.raises(RecordNotFound):
            await plane.heartbeat("missing")
        with pytest.raises(RecordNotFound):
            await plane.close_session("missing", "reason")

    asyncio.run(run())


def test_session_attribution_is_persisted_with_original_metadata() -> None:
    async def run() -> None:
        plane = control_plane(
            session_id_factory=lambda: "session-a",
            resource_policy=ResourcePolicy(scope="session", tag_keys=("project",)),
        )
        metadata = {"run_id": "run-1", "spindle.tag.project": "example", "notes": "keep"}
        session = await plane.create_session(tags=("legacy-tag",), user_metadata=metadata)
        stored = await plane.kv.get(session_key(session.session_id))
        restored = SessionRecord.model_validate(stored)
        assert restored == session
        assert restored.user_metadata == metadata
        assert restored.tags == ("legacy-tag",)
        assert restored.attribution.app_tags() == {
            "spindle_session_id": "session-a",
            "spindle_run_id": "run-1",
            "project": "example",
        }
        assert restored.attribution.resource_digest("frontend", "model") == session.attribution.resource_digest("frontend", "model")

    asyncio.run(run())


def test_invalid_attribution_does_not_create_partial_session() -> None:
    async def run() -> None:
        plane = control_plane(
            session_id_factory=lambda: "session-a",
            resource_policy=ResourcePolicy(scope="session"),
        )
        with pytest.raises(ValueError, match="not allowed"):
            await plane.create_session(user_metadata={"spindle.tag.project": "example"})
        assert await plane.kv.list_keys("session:") == ()
        assert await plane.kv.list_keys("session_last_seen:") == ()

    asyncio.run(run())


def test_shared_session_serialization_preserves_legacy_record_shape() -> None:
    async def run() -> None:
        plane = control_plane(session_id_factory=lambda: "session-a", clock=lambda: 1.0)
        session = await plane.create_session(user_metadata={"run_id": "legacy/run"})
        legacy = {
            "schema_version": 1,
            "session_id": "session-a",
            "created_at": 1.0,
            "tags": [],
            "user_metadata": {"run_id": "legacy/run"},
            "sdk_version": None,
        }
        assert await plane.kv.get(session_key("session-a")) == legacy
        assert SessionRecord.model_validate(legacy).attribution is None
        assert session.model_dump(mode="json") == legacy

    asyncio.run(run())


def test_session_record_rejects_another_sessions_attribution() -> None:
    with pytest.raises(ValueError, match="attribution must belong"):
        SessionRecord(
            session_id="session-a", created_at=1.0,
            attribution=SessionAttribution(session_id="session-b"),
        )


def test_session_compute_can_be_exposed_over_http() -> None:
    from spindle.control_plane import create_control_plane_app
    plane = control_plane(resource_policy=ResourcePolicy(scope="session"))
    assert create_control_plane_app(plane, []) is not None
