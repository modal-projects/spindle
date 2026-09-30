from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import modal

from spindle.control_plane.trainer_reconciler import TrainerPlanRecord

from .kv import shared_kv

RECONCILE_CALL_KEY = "trainer_reconcile:call"
RECONCILE_START_KEY = "trainer_reconcile:start"
RECONCILE_STARTED_AT_KEY = "trainer_reconcile:started_at"
RECONCILE_REQUEST_PREFIX = "trainer_reconcile_request:"
RECONCILE_COMPLETE_PREFIX = "trainer_reconcile_complete:"
TRAINER_PLAN_PREFIX = "trainer_plan:"
RECONCILE_DEBOUNCE_SECONDS = 5.0

_log = logging.getLogger(__name__)


def trainer_plan_key(definition_id: str) -> str:
    return f"{TRAINER_PLAN_PREFIX}{definition_id}"


async def request_reconcile(
    spawn: Callable[[float], Awaitable[str]],
    definition_id: str,
) -> str | None:
    store = shared_kv()
    await store.put(
        f"{RECONCILE_REQUEST_PREFIX}{definition_id}",
        uuid.uuid4().hex,
    )
    call_id = await store.get(RECONCILE_CALL_KEY)
    if call_id is not None and not await _call_finished(str(call_id)):
        return None
    claim = {"token": uuid.uuid4().hex, "created_at": time.time()}
    inserted = await store.put_if_absent(RECONCILE_START_KEY, claim)
    if not inserted.created:
        value = inserted.value
        if (
            isinstance(value, dict)
            and time.time() - float(value.get("created_at", 0)) < 60
        ):
            return None
        await store.delete(RECONCILE_START_KEY)
        return await request_reconcile(spawn, definition_id)
    try:
        now = time.time()
        last_started = await store.get(RECONCILE_STARTED_AT_KEY)
        delay = max(
            0.0,
            RECONCILE_DEBOUNCE_SECONDS
            - (
                now - float(last_started)
                if isinstance(last_started, (int, float))
                else RECONCILE_DEBOUNCE_SECONDS
            ),
        )
        call_id = await spawn(delay)
        await store.put(RECONCILE_CALL_KEY, call_id)
        await store.put(RECONCILE_STARTED_AT_KEY, now + delay)
        return call_id
    finally:
        await store.delete(RECONCILE_START_KEY)


async def pending_reconciliations() -> dict[str, str]:
    items = await shared_kv().list_items(
        RECONCILE_REQUEST_PREFIX,
        RECONCILE_COMPLETE_PREFIX,
    )
    requested = {
        key.removeprefix(RECONCILE_REQUEST_PREFIX): str(value)
        for key, value in items
        if key.startswith(RECONCILE_REQUEST_PREFIX)
    }
    completed = {
        key.removeprefix(RECONCILE_COMPLETE_PREFIX): str(value)
        for key, value in items
        if key.startswith(RECONCILE_COMPLETE_PREFIX)
    }
    return {
        definition_id: token
        for definition_id, token in requested.items()
        if completed.get(definition_id) != token
    }


async def complete_reconcile(definition_id: str, token: str) -> None:
    await shared_kv().put(
        f"{RECONCILE_COMPLETE_PREFIX}{definition_id}",
        token,
    )


async def release_reconcile_call(call_id: str) -> None:
    store = shared_kv()
    if await store.get(RECONCILE_CALL_KEY) == call_id:
        await store.delete(RECONCILE_CALL_KEY)


async def list_trainer_plans() -> tuple[TrainerPlanRecord, ...]:
    return tuple(
        TrainerPlanRecord.model_validate(value)
        for _, value in await shared_kv().list_items(TRAINER_PLAN_PREFIX)
    )


async def _call_finished(call_id: str) -> bool:
    try:
        await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
    except TimeoutError:
        return False
    except (
        modal.exception.AuthError,
        modal.exception.ConnectionError,
        modal.exception.InternalError,
        modal.exception.ResourceExhaustedError,
        modal.exception.ServiceError,
        OSError,
    ):
        _log.exception("trainer reconcile liveness poll %s", call_id)
        return False
    except Exception:  # noqa: BLE001 - any completed remote failure ends the call
        return True
    return True
