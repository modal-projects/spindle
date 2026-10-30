"""Miles rejects adapter slots under all-gather context parallelism. With bshd
and one row per micro-batch, each CP rank holds one contiguous chunk of the row,
so the row's adapter gets that chunk's length in tokens.
"""

from __future__ import annotations

import torch


class _SlotlessIterator:
    """Hide adapter slots from Miles' get_batch and keep them for the caller."""

    def __init__(self, inner):
        self._inner = inner
        self.slots = None

    def get_next(self, keys):
        batch = self._inner.get_next(keys)
        self.slots = batch.pop("adapter_slots", None)
        return batch

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install(megatron_model) -> None:
    original = megatron_model.get_batch
    if getattr(original, "__spindle_allgather_cp_lora__", False):
        return

    def get_batch(data_iterator, keys, *args, allgather_cp=False, **kwargs):
        qkv_format = args[1] if len(args) > 1 else kwargs.get("qkv_format", "thd")
        if not (allgather_cp and qkv_format == "bshd"):
            return original(
                data_iterator, keys, *args, allgather_cp=allgather_cp, **kwargs
            )
        wrapped = _SlotlessIterator(data_iterator)
        batch = original(wrapped, keys, *args, allgather_cp=allgather_cp, **kwargs)
        if wrapped.slots is not None:
            tokens = batch["tokens"]
            counts = torch.zeros(
                data_iterator.rollout_data["n_adapters"],
                dtype=torch.int32,
                device=tokens.device,
            )
            for slot in wrapped.slots:
                counts[slot] += tokens.size(1)
            batch["adapter_slots"] = wrapped.slots
            batch["adapter_token_counts"] = counts
        return batch

    get_batch.__spindle_allgather_cp_lora__ = True
    megatron_model.get_batch = get_batch
