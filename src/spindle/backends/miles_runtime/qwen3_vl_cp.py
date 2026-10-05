"""Hand Miles' rank-local Qwen3-VL MRoPE positions to Bridge under CP.

Miles pre-shards packed THD rows across context-parallel ranks and supplies
per-segment MRoPE positions through a patched ``get_rope_index``. The pinned
Megatron-Bridge accepts pre-sharded rows only with explicit 3D
``position_ids`` and raises before it calls ``get_rope_index``.
"""

from __future__ import annotations

from functools import wraps

_INSTALLED = "_spindle_cp_position_ids"


def install_qwen3_vl_cp_position_ids(*, bridge_model, miles_qwen3_vl) -> None:
    model_cls = bridge_model.Qwen3VLModel
    forward = model_cls.forward
    if getattr(forward, _INSTALLED, False):
        return

    @wraps(forward)
    def forward_with_positions(self, *args, **kwargs):
        if kwargs.get("position_ids") is None:
            parsed = miles_qwen3_vl._parse_packed_thd(args, kwargs)
            if miles_qwen3_vl._prepare_cp_local_context(parsed) is not None:
                kwargs["position_ids"] = miles_qwen3_vl._build_packed_positions(
                    self, parsed, kwargs, bridge_model.get_rope_index
                )
        return forward(self, *args, **kwargs)

    setattr(forward_with_positions, _INSTALLED, True)
    model_cls.forward = forward_with_positions
