from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel

from spindle.control_plane.records import (
    ModelRecord,
    PlacementRecord,
    SessionClosedRecord,
)
from spindle.providers.contracts import (
    EngineInstance,
    EnginePlatform,
    KeyValueStore,
)

from .store import typed_kv

TRAINER_PLAN_PREFIX = "trainer_plan:"

_log = logging.getLogger(__name__)


class TrainerPlanRecord(BaseModel):
    definition_id: str
    observed_at: float
    live_models: int
    desired_instances: int
    maximum_instances: int | None
    active_instances: int
    starting_instances: int
    draining_instances: int
    pending_models: int
    converged: bool
    updated_at: float


def trainer_plan_key(definition_id: str) -> str:
    return f"{TRAINER_PLAN_PREFIX}{definition_id}"


async def reconcile_trainers(
    kv: KeyValueStore,
    engines: EnginePlatform,
    definition_id: str,
    *,
    revision: str | None,
    maximum_instances: int | None,
    minimum_instances: int = 0,
    models_per_instance: int = 1,
    scale_up: bool = True,
    clock: Callable[[], float] = time.time,
) -> TrainerPlanRecord:
    if minimum_instances < 0:
        raise ValueError("minimum_instances must be non-negative")
    if maximum_instances is not None and maximum_instances < minimum_instances:
        raise ValueError("maximum_instances must be at least minimum_instances")
    if models_per_instance < 1:
        raise ValueError("models_per_instance must be positive")
    kv = typed_kv(kv)

    def current_revision(instance: EngineInstance) -> bool:
        return revision is None or instance.revision == revision

    instances = await engines.list_instances()
    items = await kv.list_items(
        "model:",
        "placement:",
        "session_closed:",
        "trainer_demand:",
    )
    closed = {
        cast(SessionClosedRecord, value).session_id
        for key, value in items
        if key.startswith("session_closed:")
    }
    models = [
        cast(ModelRecord, value)
        for key, value in items
        if key.startswith("model:")
        and cast(ModelRecord, value).engine_definition_id == definition_id
        and cast(ModelRecord, value).session_id not in closed
    ]
    placements = {
        record.model_id: record
        for key, value in items
        if key.startswith("placement:")
        for record in (cast(PlacementRecord, value),)
    }
    demand_ids = {
        str(value["model_id"])
        for key, value in items
        if key.startswith("trainer_demand:")
        and isinstance(value, dict)
        and value.get("definition_id") == definition_id
        and isinstance(value.get("model_id"), str)
    } | set(placements)
    records = [
        instance
        for instance in instances
        if instance.definition_id == definition_id and not instance.terminal
    ]
    current_ids = {record.instance_id for record in records if current_revision(record)}
    demanded_models = [model for model in models if model.model_id in demand_ids]
    live_models = [
        model
        for model in demanded_models
        if (
            model.model_id not in placements
            or placements[model.model_id].engine_instance_id in current_ids
        )
    ]
    desired = max(
        minimum_instances,
        math.ceil(len(live_models) / models_per_instance),
    )
    if maximum_instances is not None:
        desired = min(desired, maximum_instances)
    current = [record for record in records if current_revision(record)]
    usable = [record for record in current if record.state in {"starting", "running"}]
    if scale_up:
        missing = max(0, desired - len(usable))
        if (
            missing
            and maximum_instances is not None
            and len(records) >= maximum_instances
        ):
            stale = [
                record
                for record in records
                if not current_revision(record)
                and record.state in {"running", "draining"}
            ]
            await _stop_empty(
                engines,
                stale,
                len(records) - maximum_instances + missing,
            )
            records = [
                instance
                for instance in await engines.list_instances()
                if instance.definition_id == definition_id and not instance.terminal
            ]
        available = (
            missing
            if maximum_instances is None
            else max(0, maximum_instances - len(records))
        )
        for _ in range(min(max(0, desired - len(usable)), available)):
            instance = await engines.spawn_instance(definition_id)
            usable.append(instance)
    records = [
        instance
        for instance in await engines.list_instances()
        if instance.definition_id == definition_id and not instance.terminal
    ]
    current = [record for record in records if current_revision(record)]
    running = sorted(
        (record for record in current if record.state == "running"),
        key=lambda record: record.instance_id,
    )
    await _stop_empty(engines, running, max(0, len(current) - desired))
    for record in records:
        if not current_revision(record) and record.state == "running":
            await engines.set_instance_state(record.instance_id, "draining")
    draining = [
        instance
        for instance in await engines.list_instances()
        if instance.definition_id == definition_id and instance.state == "draining"
    ]
    await _stop_empty(engines, draining, len(draining))
    final = [
        instance
        for instance in await engines.list_instances()
        if instance.definition_id == definition_id and not instance.terminal
    ]
    current_final = [record for record in final if current_revision(record)]
    active = sum(record.state == "running" for record in current_final)
    starting = sum(record.state == "starting" for record in current_final)
    draining_count = sum(record.state == "draining" for record in final)
    stale_starting = any(
        record.state == "starting" and not current_revision(record) for record in final
    )
    plan = TrainerPlanRecord(
        definition_id=definition_id,
        observed_at=clock(),
        live_models=len(demanded_models),
        desired_instances=desired,
        maximum_instances=maximum_instances,
        active_instances=active,
        starting_instances=starting,
        draining_instances=draining_count,
        pending_models=(
            0
            if maximum_instances is None
            else max(
                0,
                len(live_models) - maximum_instances * models_per_instance,
            )
        ),
        converged=(
            active + starting == desired and draining_count == 0 and not stale_starting
        ),
        updated_at=clock(),
    )
    await kv.put(trainer_plan_key(definition_id), plan.model_dump(mode="json"))
    return plan


async def _model_ids(
    engines: EnginePlatform,
    records: list[EngineInstance],
) -> dict[str, tuple[str, ...]]:
    result = {}
    for record in records:
        try:
            result[record.instance_id] = await engines.client(
                record.instance_id
            ).model_ids()
        except Exception:  # noqa: BLE001 - one unreachable engine must not block others
            _log.exception("list models on %s", record.instance_id)
    return result


async def _stop_empty(
    engines: EnginePlatform,
    records: list[EngineInstance],
    limit: int,
) -> None:
    infos = await _model_ids(engines, records)
    stopped = 0
    for record in records:
        if stopped >= limit or infos.get(record.instance_id) != ():
            continue
        try:
            acknowledged = await engines.client(record.instance_id).shutdown_if_idle()
        except Exception:  # noqa: BLE001 - failed shutdowns are retried later
            _log.exception("shutdown %s", record.instance_id)
            continue
        if not acknowledged:
            continue
        await engines.stop_instance(record.instance_id)
        stopped += 1
