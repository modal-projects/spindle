from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from tinker import ForwardBackwardOutput, TensorData

from spindle.backends.contract import ForwardBatch
from spindle.backends.score_centering import (
    REPLAY_LOSSES,
    validate_loss_config,
    validate_support,
)
from .replay_data import replay_row

SUPPORTED_LOSSES = (
    frozenset({"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"})
    | REPLAY_LOSSES
)
_RL_LOSSES = SUPPORTED_LOSSES - {"cross_entropy"}


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    slot_rows: tuple[tuple[int, dict[str, Any]], ...]
    locations: tuple[tuple[int, int], ...]
    target_lengths: tuple[int, ...]


def pad_slot_rows(
    slot_rows: tuple[tuple[int, dict[str, Any]], ...],
    multiple: int,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Pad to a multiple of multiple with zero-weight rows for Miles DP sharding."""
    if multiple <= 1 or len(slot_rows) % multiple == 0:
        return slot_rows
    slot, last = slot_rows[-1]
    pad: dict[str, Any] = {
        "tokens": last["tokens"][:2],
        "target_len": 1,
        "target_tokens": last["tokens"][1:2],
    }
    for key in ("weights", "advantages", "sampling_logprobs"):
        if key in last:
            pad[key] = np.zeros(1, dtype=np.float32)
    if "routed_experts" in last:
        route = last["routed_experts"]
        width = route["shape"][1] * route["shape"][2]
        pad["routed_experts"] = {
            "values": route["values"][:width],
            "shape": [1, *route["shape"][1:]],
        }
    if "sampling_mask_ids" in last:
        pad["sampling_mask_ids"] = []
        pad["sampling_mask_offsets"] = [0, 0]
    if "sampling_mask_logprobs" in last:
        pad["sampling_mask_logprobs"] = []
    n_pad = -len(slot_rows) % multiple
    return (*slot_rows, *((slot, dict(pad)) for _ in range(n_pad)))


def prepare_batch(
    batch: ForwardBatch,
    slots: dict[str, int],
    sequence_alignment: int = 1,
) -> PreparedBatch:
    if batch.loss_fn not in SUPPORTED_LOSSES:
        raise ValueError(f"Miles does not support loss {batch.loss_fn!r}")
    if batch.loss_fn in REPLAY_LOSSES:
        validate_loss_config(batch.loss_fn, dict(batch.loss_fn_config))
    if sequence_alignment < 1:
        raise ValueError("sequence_alignment must be at least 1")
    entries: list[tuple[int, int, int, dict[str, Any], int]] = []
    for item_index, item in enumerate(batch.items):
        if item.model_id not in slots:
            raise ValueError(f"model {item.model_id} is not loaded")
        if not item.data:
            raise ValueError(f"forward_backward has no data for model {item.model_id}")
        for datum_index, datum in enumerate(item.data):
            row = _datum_row(datum, batch.loss_fn, datum_index)
            target_length = int(row["target_len"])
            _align_row(row, sequence_alignment)
            entries.append(
                (slots[item.model_id], item_index, datum_index, row, target_length)
            )

    entries.sort(key=lambda entry: entry[0])
    return PreparedBatch(
        slot_rows=tuple((slot, row) for slot, _, _, row, _ in entries),
        locations=tuple(
            (item_index, datum_index) for _, item_index, datum_index, _, _ in entries
        ),
        target_lengths=tuple(length for *_, length in entries),
    )


def _align_row(row: dict[str, Any], multiple: int) -> None:
    """Pad each sequence to a multiple of ``2 * cp * tp`` so context-parallel
    chunking and tensor-parallel sharding both divide it evenly."""

    pad = -len(row["tokens"]) % multiple
    if pad == 0:
        return
    filler = row["tokens"][-1]
    row["tokens"] = np.concatenate([row["tokens"], np.full(pad, filler)])
    targets = row["target_tokens"]
    row["target_tokens"] = np.concatenate(
        [targets, np.full((pad, *targets.shape[1:]), filler, dtype=targets.dtype)]
    )
    row["target_len"] = len(row["target_tokens"])
    for key in ("weights", "advantages", "sampling_logprobs"):
        values = row.get(key)
        if values is not None:
            row[key] = np.concatenate(
                [values, np.zeros((pad, *values.shape[1:]), dtype=values.dtype)]
            )
    if "routed_experts" in row:
        route = row["routed_experts"]
        width = route["shape"][1] * route["shape"][2]
        route["values"] = [*route["values"], *([-1] * pad * width)]
        route["shape"][0] += pad
    if "sampling_mask_offsets" in row:
        offsets = row["sampling_mask_offsets"]
        row["sampling_mask_offsets"] = [*offsets, *([offsets[-1]] * pad)]


def build_outputs(
    batch: ForwardBatch,
    prepared: PreparedBatch,
    raw_outputs: list[dict[str, Any]],
) -> tuple[ForwardBackwardOutput, ...]:
    if len(raw_outputs) != len(prepared.locations):
        raise RuntimeError(
            f"Miles returned {len(raw_outputs)} datum outputs for "
            f"{len(prepared.locations)} inputs"
        )
    grouped: list[list[dict[str, Any] | None]] = [
        [None] * len(item.data) for item in batch.items
    ]
    for location, output, target_length in zip(
        prepared.locations, raw_outputs, prepared.target_lengths, strict=True
    ):
        item_index, datum_index = location
        shape = list(
            batch.items[item_index]
            .data[datum_index]
            .loss_fn_inputs["target_tokens"]
            .shape
            or [target_length]
        )
        values = np.asarray(output["logprobs"], dtype=np.float32)[:target_length]
        grouped[item_index][datum_index] = {
            "loss": float(output["loss"]),
            "logprobs": values.reshape(-1).tolist(),
            "shape": shape,
            "diagnostics": output.get("metrics"),
        }

    results = []
    for records in grouped:
        if any(record is None for record in records):
            raise RuntimeError("Miles omitted one or more datum outputs")
        complete = [record for record in records if record is not None]
        loss = sum(record["loss"] for record in complete)
        tokens = sum(record["shape"][0] for record in complete)
        results.append(
            ForwardBackwardOutput(
                loss_fn_output_type="ArrayRecord",
                loss_fn_outputs=[
                    {
                        "loss:sum": TensorData(
                            data=[record["loss"]],
                            dtype="float32",
                            shape=[1],
                        ),
                        "logprobs": TensorData(
                            data=record["logprobs"],
                            dtype="float32",
                            shape=record["shape"],
                        ),
                    }
                    for record in complete
                ],
                metrics=_metrics(complete, loss, tokens),
            )
        )
    return tuple(results)


def _metrics(complete, loss, tokens):
    metrics = {
        "loss:sum": loss,
        "loss:mean": loss / max(tokens, 1),
        "tokens:sum": float(tokens),
        "n_sequences:sum": float(len(complete)),
        "response_length:mean": tokens / max(len(complete), 1),
    }
    diagnostics = [record["diagnostics"] for record in complete]
    if diagnostics and all(row is not None for row in diagnostics):
        trained = sum(row["tokens"] for row in diagnostics)
        metrics["tokens:sum"] = float(trained)
        for key in diagnostics[0].keys() - {"tokens"}:
            total = sum(row[key] for row in diagnostics)
            metrics[f"{key}:sum"] = total
            metrics[f"{key}:mean"] = total / max(trained, 1)
    return metrics


def _datum_row(datum, loss_fn: str, datum_index: int) -> dict[str, Any]:
    inputs = datum.loss_fn_inputs
    input_tokens = np.asarray(datum.model_input.to_ints(), dtype=np.int64)
    if not input_tokens.size:
        raise ValueError(f"datum {datum_index}: model_input cannot be empty")
    target = inputs.get("target_tokens")
    if target is None:
        raise ValueError(f"datum {datum_index}: target_tokens is required")
    shape = list(target.shape or [len(target.data)])
    if len(shape) not in (1, 2) or any(size < 1 for size in shape):
        raise ValueError(
            f"datum {datum_index}: target_tokens must have shape [sequence] or [sequence, choices]"
        )
    if shape[0] != len(input_tokens):
        raise ValueError(
            f"datum {datum_index}: target_tokens length {shape[0]} "
            f"does not match model_input length {len(input_tokens)}"
        )
    targets = _shaped_values(target, shape, np.int64, datum_index, "target_tokens")
    # The final token is a placeholder after the last input position. Labels
    # are gathered separately and must never replace the actual model inputs.
    final_target = targets[-1, 0] if len(shape) == 2 else targets[-1]
    row: dict[str, Any] = {
        "tokens": np.append(input_tokens, final_target),
        "target_len": shape[0],
        "target_tokens": targets,
    }
    # Miles derives batch fields from its first datum. Forward only the fields
    # consumed by this loss so optional, unrelated inputs on one client cannot
    # cause missing-key failures when its rows are combined with another client.
    if loss_fn == "cross_entropy":
        fields = (("weights", "weights"),)
    elif loss_fn in REPLAY_LOSSES:
        fields = (
            ("weights", "weights"),
            ("advantages", "advantages"),
            ("logprobs", "sampling_logprobs"),
        )
    else:
        fields = (("advantages", "advantages"), ("logprobs", "sampling_logprobs"))
    for source, destination in fields:
        value = inputs.get(source)
        if value is not None:
            row[destination] = _shaped_values(
                value, shape, np.float32, datum_index, source
            )

    if loss_fn == "cross_entropy":
        row.setdefault("weights", np.ones(shape, dtype=np.float32))
    elif loss_fn in _RL_LOSSES:
        missing = [
            name for name in ("advantages", "sampling_logprobs") if name not in row
        ]
        if missing:
            raise ValueError(
                f"datum {datum_index}: {', '.join(missing)} required for {loss_fn}"
            )
    if loss_fn in REPLAY_LOSSES:
        if len(shape) != 1:
            raise ValueError(
                f"datum {datum_index}: {loss_fn} requires one target per position"
            )
        weights = row.get("weights")
        validate_support(
            inputs,
            [int(value) for value in targets],
            [1.0] * len(targets)
            if weights is None
            else [float(value) for value in weights],
            centered=loss_fn != "mis",
        )
    row.update(replay_row(inputs, targets))
    return row


def _shaped_values(tensor, shape, dtype, datum_index, name) -> np.ndarray:
    values = _tensor_values(tensor)
    actual_shape = list(tensor.shape or [values.size])
    if actual_shape != shape or values.size != prod(shape):
        raise ValueError(
            f"datum {datum_index}: {name} must have shape {shape} with matching data"
        )
    return values.astype(dtype).reshape(shape)


def _tensor_values(tensor) -> np.ndarray:
    """Return row-major values; the SDK sends mostly-zero 2-D tensors as CSR."""
    values = np.asarray(tensor.data)
    crow = tensor.sparse_crow_indices
    if crow is None:
        return values.reshape(-1)
    shape = list(tensor.shape or [])
    columns = tensor.sparse_col_indices
    rows, width = (1, shape[0]) if len(shape) == 1 else (shape + [None, None])[:2]
    crow = np.asarray(crow, dtype=np.int64)
    if (
        len(shape) not in (1, 2)
        or columns is None
        or len(crow) != rows + 1
        or crow[0] != 0
        or crow[-1] != len(values)
        or len(columns) != len(values)
        or (np.diff(crow) < 0).any()
    ):
        raise ValueError(
            "CSR TensorData must have shape [sequence] or [sequence, choices] "
            "with row pointers covering its values"
        )
    columns = np.asarray(columns, dtype=np.int64)
    if ((columns < 0) | (columns >= width)).any():
        raise ValueError("CSR TensorData column indices must be within the row width")
    dense = np.zeros(rows * width, dtype=values.dtype)
    row_ids = np.repeat(np.arange(rows), np.diff(crow))
    np.add.at(dense, row_ids * width + columns, values)
    return dense
