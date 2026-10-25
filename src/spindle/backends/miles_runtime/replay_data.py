"""Validate replay tensors before admitting work to distributed execution."""

from __future__ import annotations

from functools import wraps
from itertools import pairwise
from math import isfinite, prod

import numpy as np

from spindle.backends.score_centering import REPLAY_LOSSES
from spindle.replay import REPLAY_FIELDS

_replay_builder = None


def install_distribution_sharding(conversion):
    original = conversion._package_shards
    if getattr(original, "__spindle_sampling_probabilities__", False):
        return

    @wraps(original)
    def package(args, data, partitions):
        shards = original(args, data, partitions)
        key = "rollout_sampling_mask_logprobs"
        if key in data:
            for shard, partition in zip(shards, partitions, strict=True):
                shard[key] = [data[key][index] for index in partition]
        return shards

    package.__spindle_sampling_probabilities__ = True
    conversion._package_shards = package


def replay_row(inputs, targets: list[int]) -> dict:
    row = {}
    n = len(targets)
    for name in (REPLAY_FIELDS - {"sampling_mask_logprobs"}) & inputs.keys():
        tensor = inputs[name]
        values = list(tensor.data)
        shape = list(tensor.shape or ())
        if (
            tensor.dtype != "int64"
            or tensor.sparse_crow_indices is not None
            or any(type(v) is not int for v in values)
            or prod(shape) != len(values)
        ):
            raise ValueError(
                f"{name} must be a dense int64 tensor with a matching shape"
            )
        if any(
            v < (-1 if name == "routed_experts" else 0) or v > 2**31 - 1 for v in values
        ):
            raise ValueError(f"{name} values are outside the supported int32 range")
        if name == "routed_experts":
            if len(shape) != 3 or shape[0] != n or min(shape[1:]) < 1:
                raise ValueError(
                    "routed_experts must have shape [input_tokens, layers, experts_per_token]"
                )
            row[name] = {"values": values, "shape": shape}
        else:
            if shape != [len(values)]:
                raise ValueError(f"{name} must be one-dimensional")
            row[name] = values
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
            or any(b < a for a, b in pairwise(offsets))
        ):
            raise ValueError(
                "sampling mask offsets must have input_tokens + 1 entries, start at zero, and end at the id count"
            )
        for token, a, b in zip(targets, offsets, offsets[1:], strict=False):
            if a != b and (token not in ids[a:b] or len(set(ids[a:b])) != b - a):
                raise ValueError(
                    "each sampling support must contain its target and have no duplicate ids"
                )
    if "sampling_mask_logprobs" in inputs:
        tensor = inputs["sampling_mask_logprobs"]
        values = list(tensor.data)
        if (
            ids is None
            or tensor.dtype != "float32"
            or tensor.sparse_crow_indices is not None
            or list(tensor.shape or ()) != [len(ids)]
            or len(values) != len(ids)
            or any(not isfinite(v) or v > 0 for v in values)
        ):
            raise ValueError(
                "sampling_mask_logprobs must be dense float32 values aligned with support IDs"
            )
        row["sampling_mask_logprobs"] = values
    return row


def validate_routed_experts(
    slot_rows, *, num_layers: int, num_experts: int | None, topk: int
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
        values = routes["values"]
        if any(expert >= num_experts for expert in values):
            raise ValueError(
                f"routed_experts IDs must be below the model's expert count ({num_experts})"
            )
        for offset in range(0, len(values), topk):
            experts = values[offset : offset + topk]
            if all(expert == -1 for expert in experts):
                continue
            if -1 in experts:
                raise ValueError(
                    "routed_experts padding must fill the entire top-k row with -1"
                )
            if len(set(experts)) != topk:
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
    if any("sampling_mask_logprobs" in d for d in datums):
        train_data["rollout_sampling_mask_logprobs"] = [
            np.asarray(d.get("sampling_mask_logprobs", []), dtype=np.float32)
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
    runtime.ROLLOUT_DATA_VALUE_SPEC["rollout_sampling_mask_logprobs"] = (
        runtime.ROLLOUT_DATA_VALUE_SPEC["rollout_sampling_mask_ids"]
    )
    original_execute = runtime.MilesBackend._execute_batch

    @wraps(original_execute)
    async def execute(self, method, batch_id, slot_datums, loss_fn, loss_fn_config):
        if loss_fn not in REPLAY_LOSSES:
            return await original_execute(
                self, method, batch_id, slot_datums, loss_fn, loss_fn_config
            )
        train_data = runtime._build_train_data(
            runtime._pad_to_dp_multiple(slot_datums, self.dp_size)
        )
        train_data.update(loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        results = await self._call_trainer(method, batch_id, train_data)
        by_index = {}
        for result in results:
            if "error" in result:
                return result
            for row in result["per_datum"]:
                index = int(row["sample_index"])
                if index not in by_index:
                    by_index[index] = {
                        "loss": float(row["loss"]),
                        "logprobs": row["logprobs"].tolist(),
                        "metrics": row["metrics"],
                    }
        return [by_index[index] for index in range(len(slot_datums))]

    runtime.MilesBackend._execute_batch = execute
