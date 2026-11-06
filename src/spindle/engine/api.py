from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from tinker import AdamParams

if TYPE_CHECKING:
    from .operations import OperationPayload


class OperationKind(StrEnum):
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS = "save_weights"
    LOAD_WEIGHTS = "load_weights"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    SKIP = "skip"


BATCHED_OPERATIONS = frozenset({OperationKind.FORWARD, OperationKind.FORWARD_BACKWARD})

JSON_OPERATIONS = (
    OperationKind.FORWARD,
    OperationKind.OPTIM_STEP,
    OperationKind.SAVE_WEIGHTS,
    OperationKind.LOAD_WEIGHTS,
    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
)


class FutureStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class FutureState:
    status: FutureStatus
    result: object | None = None
    error: str | None = None


@dataclass(frozen=True)
class Command:
    """A model operation passed from the engine to its executor."""

    model_id: str
    kind: OperationKind
    payload: OperationPayload
    encoded_payload: bytes | None = field(default=None, compare=False, repr=False)


class EngineApi(Protocol):
    async def accept_model(self, model_id: str, spec: object) -> bool: ...

    async def model_ids(self) -> tuple[str, ...]: ...

    async def forward_backward(self, body: bytes, content_type: str) -> str: ...

    async def forward(self, request: dict) -> str: ...

    async def optim_step(self, request: dict) -> str: ...

    async def save_weights(self, request: dict) -> str: ...

    async def load_weights(self, request: dict) -> str: ...

    async def save_weights_for_sampler(self, request: dict) -> str: ...

    async def skip_sequence(self, model_id: str, seq_id: int, error: str) -> str: ...

    async def retrieve_future(
        self,
        request_id: str,
        timeout: float = 0.0,
    ) -> FutureState | None: ...

    async def unload_model(self, model_id: str) -> None: ...

    async def shutdown_if_idle(self) -> bool: ...


async def submit_json_operation(
    engine: EngineApi,
    kind: OperationKind,
    request: dict,
) -> str:
    match kind:
        case OperationKind.FORWARD:
            return await engine.forward(request)
        case OperationKind.OPTIM_STEP:
            return await engine.optim_step(request)
        case OperationKind.SAVE_WEIGHTS:
            return await engine.save_weights(request)
        case OperationKind.LOAD_WEIGHTS:
            return await engine.load_weights(request)
        case OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            return await engine.save_weights_for_sampler(request)
        case _:
            raise ValueError(f"{kind.value} is not a JSON operation")


FAILED_COMMAND = "failed_command"


def command_failure(error: str) -> dict[str, str]:
    return {FAILED_COMMAND: error}


def command_error(result: object) -> str | None:
    if isinstance(result, dict) and isinstance(result.get(FAILED_COMMAND), str):
        return result[FAILED_COMMAND]
    return None


def split_optimizer_group(
    executions: tuple[Command, ...],
) -> tuple[tuple[Command, ...], tuple[Command, ...]]:
    """Split a group into its leading optimizer steps and the forward batch."""
    optimizers = tuple(
        item for item in executions if item.kind == OperationKind.OPTIM_STEP
    )
    forwards = executions[len(optimizers) :]
    if not optimizers or len(optimizers) + len(forwards) < 2:
        raise ValueError(
            "optimizer group requires optimizer steps and forward commands"
        )
    if any(item.kind == OperationKind.OPTIM_STEP for item in forwards):
        raise ValueError("optimizer steps must lead the group")
    if any(not isinstance(item.payload, AdamParams) for item in optimizers):
        raise TypeError("optim_step requires an optimizer payload")
    model_ids = [item.model_id for item in optimizers]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("optimizer group contains duplicate adapters")
    if any(item.model_id in model_ids for item in forwards):
        raise ValueError("optimizer and forward batch must use different adapters")
    return optimizers, forwards


class Executor(Protocol):
    async def accept_model(self, model_id: str, spec: object) -> None: ...

    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object: ...

    async def execute_forward_backward_batch(
        self,
        executions: tuple[Command, ...],
    ) -> tuple[object, ...]: ...

    async def execute_optimizer_group(
        self, executions: tuple[Command, ...]
    ) -> tuple[object, ...]:
        """Run the leading optimizer steps and the disjoint forward batch in
        one command; a slot's failure is returned with ``command_failure``."""
        ...

    async def capture_snapshot(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        """Capture immutable state for later checkpoint or sampler persistence."""
        ...

    async def persist_snapshot(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
        capture: object,
    ) -> object:
        """Persist captured state, also publishing it for sampler operations."""
        ...

    async def unload_model(self, model_id: str) -> None: ...

    async def close(self) -> None: ...
