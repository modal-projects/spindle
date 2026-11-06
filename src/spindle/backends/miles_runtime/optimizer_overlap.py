"""Stage slot updates while another adapter uses shared model buffers."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.distributed

STREAM_POOL_ENV = "SPINDLE_OPTIMIZER_STREAMS"
DEFAULT_STREAM_POOL = 4
REMOTE_FAILURE = "optimizer step failed on another rank"


@dataclass
class DetachedSlotState:
    weights: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )
    gradients: list[torch.Tensor] = field(default_factory=list)

    def commit(self, slot: int):
        for parameter, updated in self.weights[slot]:
            parameter.data.copy_(updated)


@contextmanager
def detached_slot_state(parameters_by_slot):
    saved = []
    detached = DetachedSlotState()
    try:
        for slot, parameters in parameters_by_slot.items():
            staged = detached.weights[slot] = []
            for parameter in parameters:
                main_grad = getattr(parameter, "main_grad", None)
                grad = parameter.grad
                weight = parameter.data
                saved.append((parameter, main_grad, grad, weight))
                updated = weight.clone()
                staged.append((parameter, updated))
                parameter.data = updated
                copied_main = None if main_grad is None else main_grad.clone()
                copied_grad = None if grad is None else grad.clone()
                detached.gradients.extend(
                    value for value in (copied_main, copied_grad) if value is not None
                )
                if main_grad is not None:
                    main_grad.zero_()
                    parameter.main_grad = copied_main
                if grad is not None:
                    grad.zero_()
                parameter.grad = copied_grad
        yield detached
    finally:
        for parameter, main_grad, grad, weight in saved:
            parameter.data = weight
            if main_grad is not None:
                parameter.main_grad = main_grad
            parameter.grad = grad


def stream_pool(actor) -> tuple:
    streams = getattr(actor, "_spindle_optimizer_streams", None)
    if streams is None:
        size = max(1, int(os.environ.get(STREAM_POOL_ENV, DEFAULT_STREAM_POOL)))
        streams = tuple(torch.cuda.Stream() for _ in range(size))
        actor._spindle_optimizer_streams = streams
    return streams


def step_slot(optimizer, adam_params: dict) -> dict:
    try:
        optimizer.apply_adam_params(adam_params)
        optimizer.prepare_grads()
        outcome = optimizer.clip_and_step(adam_params["grad_clip_norm"])
        optimizer.zero_grads()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return outcome


def step_many_of(optimizers) -> object | None:
    """``step_many`` of the optimizers' shared class; None selects the per-slot loop."""
    classes = {type(optimizer) for optimizer in optimizers}
    if len(classes) != 1:
        return None
    return getattr(next(iter(classes)), "step_many", None)


def step_slots_together(step_many, requests) -> list[dict]:
    """``step_many(requests)`` takes ``(optimizer, adam_params)`` pairs in ascending
    slot order and returns one outcome per pair; an exception fails every slot."""
    try:
        outcomes = list(step_many(requests))
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}] * len(requests)
    if len(outcomes) != len(requests):
        return [{"error": "step_many returned the wrong number of outcomes"}] * len(
            requests
        )
    return outcomes


def failed_on_any_rank(failed: bool) -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return failed
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    flag = torch.tensor([int(failed)], device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def optim_step_group(
    actor,
    *,
    adam_params_by_slot,
    batch_id=None,
    rollout_data_ref=None,
    forward_only=False,
):
    """Step every slot in ``adam_params_by_slot`` while the forward batch runs.

    Slots whose optimizers share a class with ``step_many`` are stepped by one
    call on the first side stream; otherwise each slot is stepped on its own
    side stream. Every collective is issued from this thread in ascending slot
    order, which is the same on every rank.
    """
    # GPU-only Miles imports are unavailable to CPU tests importing this helper.
    from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

    slots = sorted(adam_params_by_slot)
    streams = stream_pool(actor)
    compute = torch.cuda.current_stream()
    outcomes = {}
    result = {}
    parameters_by_slot = {
        slot: adapter_slot_parameters(actor.model, slot) for slot in slots
    }
    with detached_slot_state(parameters_by_slot) as detached:
        try:
            step_many = step_many_of([actor.slot_optimizers[slot] for slot in slots])
            if step_many is not None:
                streams[0].wait_stream(compute)
                with torch.cuda.stream(streams[0]):
                    batched = step_slots_together(
                        step_many,
                        [
                            (actor.slot_optimizers[slot], adam_params_by_slot[slot])
                            for slot in slots
                        ],
                    )
                outcomes = dict(zip(slots, batched, strict=True))
                if failed_on_any_rank(any("error" in item for item in batched)):
                    outcomes = {
                        slot: item if "error" in item else {"error": REMOTE_FAILURE}
                        for slot, item in outcomes.items()
                    }
            for index, slot in enumerate(slots if step_many is None else ()):
                stream = streams[index % len(streams)]
                stream.wait_stream(compute)
                with torch.cuda.stream(stream):
                    outcome = step_slot(
                        actor.slot_optimizers[slot], adam_params_by_slot[slot]
                    )
                if failed_on_any_rank("error" in outcome) and "error" not in outcome:
                    outcome = {"error": REMOTE_FAILURE}
                outcomes[slot] = outcome
            if rollout_data_ref is not None:
                method = actor.forward_only if forward_only else actor.forward_backward
                result = method(batch_id=batch_id, rollout_data_ref=rollout_data_ref)
        finally:
            # Keep detached tensors alive and drain GPU work before lifecycle operations.
            for stream in streams:
                stream.synchronize()
    for slot in slots:
        if "error" not in outcomes[slot]:
            detached.commit(slot)
    for slot in slots:
        if "grad_norm" in outcomes[slot]:
            actor.slot_optimizers[slot].allgather_params()
    result["optimizer_outcomes"] = outcomes
    return result
