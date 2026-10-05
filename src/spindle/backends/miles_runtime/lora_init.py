"""Initialize grouped-expert LoRA A like PEFT does for each expert's matrix.

PEFT (and Tinker) draw every LoRA A from ``kaiming_uniform_(a=sqrt(5))`` on the
logical ``[rank, in_features]`` matrix, i.e. ``U(-1/sqrt(in), 1/sqrt(in))``.
Bridge draws grouped-expert A on the packed ``[experts, rank, in]`` tensor, for
which torch's fan-in is ``rank * in``, so the bound is ``sqrt(rank)`` too small.
Rescaling the draw in place fixes the bound without consuming RNG, so Bridge's
RNG-tracker discipline (identical expert-data-parallel replicas) is unchanged.
"""

from __future__ import annotations

import math
from functools import wraps

from torch.nn.init import _calculate_fan_in_and_fan_out


def install_expert_lora_a_init(
    *, multi_lora, multi_lora_layers, parallel_state
) -> None:
    grouped = multi_lora_layers.MultiLoRAGroupedExpertLinear
    reset = grouped.reset_adapter
    if getattr(reset, "__spindle_peft_expert_init__", False):
        return

    def rescale(module, idx: int) -> None:
        if module._column_init_method != "kaiming":
            return
        weight = module.adapters[idx].linear_in.weight
        drawn_fan_in, _ = _calculate_fan_in_and_fan_out(weight)
        fan_in = weight.shape[-1]
        if module.input_is_parallel:
            fan_in *= parallel_state.get_expert_tensor_parallel_world_size()
        weight.data.mul_(math.sqrt(drawn_fan_in / fan_in))

    @wraps(reset)
    def reset_adapter(self, idx: int) -> None:
        reset(self, idx)
        rescale(self, idx)

    reset_adapter.__spindle_peft_expert_init__ = True
    grouped.reset_adapter = reset_adapter

    apply = multi_lora.MultiLoRA.__call__

    @wraps(apply)
    def __call__(self, model, training: bool = True):
        model = apply(self, model, training=training)
        for module in multi_lora_layers._iter_multi_lora_modules(model):
            if isinstance(module, grouped):
                for idx in range(len(module.adapters)):
                    rescale(module, idx)
        return model

    multi_lora.MultiLoRA.__call__ = __call__
