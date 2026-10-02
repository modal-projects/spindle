"""Workarounds for grouped-expert LoRA in the pinned Miles / Megatron-Bridge.

Enabled by ``EXPERT_LORA_COMPAT=1`` (set by the gpt-oss presets) and
installed from ``SpindleMilesTrainRayActor.load_slot`` / ``export_slot``.
Delete this module and those two calls once upstream ships both fixes:

* Export: the gpt-oss bridge emits each expert LoRA tensor once, from the
  writer's own EP shard, with a leading singleton axis (``(1, E_local, ...)``).
  SGLang ignores that shape, so the sampler serves no expert LoRA. The fixed
  publisher writes ``(E, ...)`` gathered over the EP group.
* Load: Miles ``load_slot`` drops the result of ``dist_checkpointing.load``.
  Tensors behind a ``ShardedTensorFactory`` (the grouped-expert SwiGLU
  ``linear_fc1`` LoRA B) are loaded into temporary copies, so the model keeps
  its init value until an optimizer step copies the FP32 masters back.

Both fixes check the upstream result first and only log when it is already
correct.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.megatron_utils.lora.slots import slice_lora_to_rank

ENV_FLAG = "EXPERT_LORA_COMPAT"

_EXPERT_MODULE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.linear_fc([12])$")
_EXPERT_HF_NAME = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)\.(lora_A|lora_B)\.weight$"
)
_HF_PROJECTION = {"1": "gate_up_proj", "2": "down_proj"}
_ADAPTER_FILE = "adapter_model.safetensors"

ExpertKey = tuple[int, str, str]


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def _log(action: str, **fields) -> None:
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"spindle_expert_lora_compat action={action} {rendered}", flush=True)


# ---------------------------------------------------------------- load


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


# ---------------------------------------------------------------- export


@contextmanager
def publishing_all_experts(publisher, model):
    """Patch ``publisher.write_adapter`` to publish every expert's LoRA."""
    original = publisher.write_adapter

    def write_adapter(adapter, path):
        original(adapter, path)
        if adapter is not None:
            _rewrite_expert_tensors(model, adapter, Path(path))

    publisher.write_adapter = write_adapter
    try:
        yield
    finally:
        del publisher.write_adapter


@torch.no_grad()
def _rewrite_expert_tensors(model, adapter, path: Path) -> None:
    gathered = _gather_expert_lora(model, adapter.slot)
    if dist.get_rank() != 0:
        return
    gathered = {
        key: slice_lora_to_rank(key[2], tensor, adapter.rank)
        for key, tensor in gathered.items()
    }
    file = path / _ADAPTER_FILE
    published = safetensors.torch.load_file(file)
    fixed, replaced = full_expert_tensors(published, gathered)
    if replaced:
        safetensors.torch.save_file(fixed, file)
    _log(
        "publish" if replaced else "publish_noop",
        slot=adapter.slot,
        replaced=replaced,
        experts=next(iter(gathered.values())).shape[0] if gathered else 0,
    )


def _gather_expert_lora(model, slot: int) -> dict[ExpertKey, torch.Tensor]:
    """All-gather each grouped-expert LoRA tensor of ``slot`` over the EP group.

    Collective; only rank 0 gets the gathered tensors, on CPU.
    """
    if mpu.get_pipeline_model_parallel_world_size() != 1:
        raise NotImplementedError("expert LoRA publish workaround assumes PP=1")
    if mpu.get_expert_tensor_parallel_world_size() != 1:
        raise NotImplementedError("expert LoRA publish workaround assumes expert TP=1")
    group = mpu.get_expert_model_parallel_group()
    world = mpu.get_expert_model_parallel_world_size()
    gathered: dict[ExpertKey, torch.Tensor] = {}
    for chunk in model:
        for name, module in chunk.named_modules():
            match = _EXPERT_MODULE.search(name)
            if match is None or not hasattr(module, "adapters"):
                continue
            layer, projection = int(match.group(1)), _HF_PROJECTION[match.group(2)]
            adapter = module.adapters[slot]
            for side, weight in (
                ("lora_A", adapter.linear_in.weight),
                ("lora_B", adapter.linear_out.weight),
            ):
                local = weight.detach().contiguous()
                parts = [torch.empty_like(local) for _ in range(world)]
                dist.all_gather(parts, local, group=group)
                if dist.get_rank() == 0:
                    gathered[(layer, projection, side)] = torch.cat(parts).cpu()
    return gathered


def full_expert_tensors(
    published: dict[str, torch.Tensor], gathered: dict[ExpertKey, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int]:
    """Replace published expert LoRA tensors with the all-expert ``(E, ...)`` form.

    The published tensor must be a prefix of the gathered experts (the writer's
    own shard), optionally with a leading singleton axis; anything else is a
    layout this workaround does not understand. Returns the new tensors and how
    many were replaced; zero means the export was already complete.
    """
    fixed = dict(published)
    replaced = 0
    seen: set[ExpertKey] = set()
    for name, tensor in published.items():
        match = _EXPERT_HF_NAME.search(name)
        if match is None:
            continue
        key = (int(match.group(1)), match.group(2), match.group(3))
        seen.add(key)
        full = gathered[key].to(tensor.dtype)
        local = (
            tensor[0]
            if tensor.ndim == full.ndim + 1 and tensor.shape[0] == 1
            else tensor
        )
        if local.ndim != full.ndim or local.shape[1:] != full.shape[1:]:
            raise ValueError(
                f"{name}: published {tuple(tensor.shape)} vs experts {tuple(full.shape)}"
            )
        if not torch.equal(local, full[: local.shape[0]]):
            raise ValueError(
                f"{name}: published experts do not match the trainer's first experts"
            )
        if tensor.shape == full.shape:
            continue
        fixed[name] = full.contiguous()
        replaced += 1
    missing = set(gathered) - seen
    if missing:
        raise ValueError(
            f"export is missing expert LoRA tensors: {sorted(missing)[:4]}"
        )
    return fixed, replaced
