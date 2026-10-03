"""Workaround for grouped-expert LoRA checkpoint load in the pinned Miles.

Enabled by ``EXPERT_LORA_COMPAT=1`` (set by the gpt-oss presets) and
installed from ``SpindleMilesTrainRayActor.load_slot``. Delete this module and
that call once Miles applies the weights returned by ``dist_checkpointing.load``.

Miles ``load_slot`` drops the result of ``dist_checkpointing.load``. Tensors
behind a ``ShardedTensorFactory`` (the grouped-expert SwiGLU ``linear_fc1``
LoRA B) are loaded into temporary copies, so the model keeps its init value
until an optimizer step copies the FP32 masters back. The restore checks the
upstream result first and only logs when it is already correct.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from miles.backends.megatron_utils.lora import checkpoint

ENV_FLAG = "EXPERT_LORA_COMPAT"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def _log(action: str, **fields) -> None:
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"spindle_expert_lora_compat action={action} {rendered}", flush=True)


@torch.no_grad()
def restore_factory_weights(
    model, slot_optimizer, path: str, *, load_optimizer: bool
) -> None:
    """Copy factory-merged checkpoint tensors into the slot's model params.

    Collective: every rank must call it after Miles ``load_slot``.
    """
    slot = slot_optimizer.slot
    weights = checkpoint._slot_weights_sharded_state_dict(model, slot)
    factories = {
        key: value
        for key, value in weights.items()
        if isinstance(value, ShardedTensorFactory)
    }
    if not factories:
        return
    targets = {key: factory.data for key, factory in factories.items()}
    shells = {checkpoint._WEIGHTS_KEY: factories}
    checkpoint._canonicalize_slot_keys(shells, slot)
    loaded = dist_checkpointing.load(shells, str(Path(path)))[checkpoint._WEIGHTS_KEY]
    max_gap = copy_loaded_into_params(targets, loaded)
    if not load_optimizer:
        slot_optimizer.reload_masters()
    _log(
        "restore" if max_gap > 0 else "restore_noop",
        rank=dist.get_rank(),
        slot=slot,
        tensors=len(targets),
        max_gap=f"{max_gap:.3e}",
    )


def copy_loaded_into_params(
    targets: dict[str, torch.Tensor], loaded: dict[str, torch.Tensor]
) -> float:
    """Copy ``loaded[key]`` into ``targets[key]``; return the largest prior gap."""
    max_gap = 0.0
    for key, param in targets.items():
        value = loaded[key]
        if value.shape != param.shape:
            raise ValueError(
                f"loaded {key} has shape {tuple(value.shape)}, param {tuple(param.shape)}"
            )
        value = value.to(device=param.device, dtype=param.dtype)
        max_gap = max(max_gap, float((value.float() - param.float()).abs().max()))
        param.copy_(value)
    return max_gap
