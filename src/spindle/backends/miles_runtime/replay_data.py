"""Validate replay tensors before admitting work to distributed execution."""

from __future__ import annotations

from math import prod

import numpy as np

from spindle.replay import REPLAY_FIELDS

_replay_builder = None


def replay_row(inputs, targets: np.ndarray) -> dict:
    row = {}
    n = len(targets)
    for name in REPLAY_FIELDS & inputs.keys():
        tensor = inputs[name]
        shape = list(tensor.shape or ())
        # TensorData.data materializes a Python list on every access. Keep the
        # SDK array throughout admission; older/list-backed inputs still work.
        if tensor.dtype != "int64" or tensor.sparse_crow_indices is not None:
            raise ValueError(
                f"{name} must be a dense int64 tensor with a matching shape"
            )
        if hasattr(tensor, "to_numpy"):
            values = np.asarray(tensor.to_numpy()).reshape(-1)
        else:
            data = tensor.data
            if any(type(value) is not int for value in data):
                raise ValueError(
                    f"{name} must be a dense int64 tensor with a matching shape"
                )
            values = np.asarray(data).reshape(-1)
        integral = not values.size or values.dtype.kind in "iu"
        if not integral or prod(shape) != values.size:
            raise ValueError(
                f"{name} must be a dense int64 tensor with a matching shape"
            )
        if values.size and (
            values.min() < (-1 if name == "routed_experts" else 0)
            or values.max() > 2**31 - 1
        ):
            raise ValueError(f"{name} values are outside the supported int32 range")
        if name == "routed_experts":
            if len(shape) != 3 or shape[0] != n or min(shape[1:]) < 1:
                raise ValueError(
                    "routed_experts must have shape [input_tokens, layers, experts_per_token]"
                )
            # Narrow only after range validation, with an owned snapshot of
            # caller data. Miles consumes int32 routes directly.
            row[name] = {"values": values.astype(np.int32), "shape": shape}
        else:
            if shape != [len(values)]:
                raise ValueError(f"{name} must be one-dimensional")
            row[name] = values.tolist()
    ids, offsets = row.get("sampling_mask_ids"), row.get("sampling_mask_offsets")
    if (ids is None) != (offsets is None):
        raise ValueError(
            "sampling_mask_ids and sampling_mask_offsets must be supplied together"
        )
    if offsets is not None:
        if (
            len(offsets) != n + 1
            or offsets[0] != 0
            or offsets[-1] != len(ids)
            or any(b < a for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError(
                "sampling mask offsets must have input_tokens + 1 entries, start at zero, and end at the id count"
            )
        for token, a, b in zip(targets, offsets, offsets[1:], strict=False):
            candidates = np.atleast_1d(token).tolist()
            if a != b and (
                any(candidate not in ids[a:b] for candidate in candidates)
                or len(set(ids[a:b])) != b - a
            ):
                raise ValueError(
                    "each sampling support must contain its target and have no duplicate ids"
                )
    return row


def validate_routed_experts(
    slot_rows, *, num_layers: int, num_experts: int | None, topk: int, moe_layer_freq=1
) -> None:
    """Check model-specific route constraints before dispatching to any GPU rank.

    replay_row owns tensor encoding and token alignment. Here the resolved Miles
    model dimensions determine which layer streams and expert IDs are legal.
    """
    for _, datum in slot_rows:
        if "routed_experts" not in datum:
            continue
        if num_experts is None or num_experts <= 0:
            raise ValueError("routed_experts requires an MoE model")
        routes = datum["routed_experts"]
        if routes["shape"][1:] != [num_layers, topk]:
            raise ValueError(
                f"routed_experts must have {num_layers} layers and {topk} experts per token"
            )
        if moe_layer_freq is None:
            moe_layer_freq = 1
        if isinstance(moe_layer_freq, int):
            if moe_layer_freq < 1:
                raise ValueError("moe_layer_freq must be positive")
            active_layers = np.arange(num_layers) % moe_layer_freq == 0
        else:
            active_layers = np.asarray(moe_layer_freq, dtype=bool)
            if active_layers.shape != (num_layers,):
                raise ValueError("moe_layer_freq must have num_layers entries")
        if not active_layers.any():
            raise ValueError("routed_experts requires an MoE model")
        # SGLang leaves dense-layer slots zero-filled. They are not routes, and
        # Miles selects only MoE streams using this same model layer layout.
        # Validate one datum at a time to bound temporary memory on long batches.
        values = np.asarray(routes["values"]).reshape(routes["shape"])
        values = values[:, active_layers, :]
        if np.any(values >= num_experts):
            raise ValueError(
                f"routed_experts IDs must be below the model's expert count ({num_experts})"
            )
        padding = values == -1
        fully_padded = padding.all(axis=-1)
        if np.any(padding.any(axis=-1) & ~fully_padded):
            raise ValueError(
                "routed_experts padding must fill the entire top-k row with -1"
            )
        sorted_values = np.sort(values, axis=-1)
        duplicates = (sorted_values[..., 1:] == sorted_values[..., :-1]).any(axis=-1)
        if np.any(duplicates & ~fully_padded):
            raise ValueError(
                "routed_experts IDs must be distinct within each top-k row"
            )


def add_replay_to_train_data(train_data: dict, datums: list[dict]) -> None:
    routes = [datum.get("routed_experts") for datum in datums]
    if any(r is not None for r in routes):
        if not all(r is not None for r in routes):
            raise ValueError(
                "router replay must be supplied for every datum in a batch"
            )
        train_data["rollout_routed_experts"] = [
            np.asarray(r["values"], dtype=np.int32).reshape(r["shape"]) for r in routes
        ]
    if any("sampling_mask_ids" in d for d in datums):
        train_data["rollout_sampling_mask_ids"] = [
            np.asarray(d.get("sampling_mask_ids", []), dtype=np.int32) for d in datums
        ]
        train_data["rollout_sampling_mask_offsets"] = [
            np.asarray(
                d.get("sampling_mask_offsets", [0] * (d["target_len"] + 1)),
                dtype=np.int64,
            )
            for d in datums
        ]


def install_bridge_replay(runtime) -> None:
    global _replay_builder
    original = runtime._build_train_data
    if original is _replay_builder:
        return

    def build(slot_datums):
        result = original(slot_datums)
        add_replay_to_train_data(result, [datum for _, datum in slot_datums])
        return result

    _replay_builder = build
    runtime._build_train_data = build
