"""Encode completed results for SDKs that request protobuf responses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import prod
from typing import Any

import numpy as np
from tinker.proto import tinker_public_pb2 as pb

PROTOBUF_MEDIA_TYPE = "application/x-protobuf"


@dataclass(frozen=True, slots=True)
class EncodedResult:
    """A completed result already in the SDK's protobuf wire format.

    Forward/backward outputs carry one value per token. The backend encodes them
    once; the engine and frontend forward the bytes to clients that accept
    protobuf and decode them only for JSON clients.
    """

    body: bytes

    def to_json(self) -> dict[str, Any]:
        return decode_forward_backward(self.body)


_DTYPES = {"float32": ("<f4", pb.DTYPE_FLOAT32), "int64": ("<i8", pb.DTYPE_INT64)}
_NUMPY_DTYPES = {pb.DTYPE_FLOAT32: ("<f4", "float32"), pb.DTYPE_INT64: ("<i8", "int64")}


def encode_forward_backward_exact(result: object) -> EncodedResult | None:
    """Encode a forward/backward result if protobuf preserves its JSON form exactly.

    That holds for dense tensors whose float values are float32 and whose integer
    values fit int64. Anything else returns None and stays JSON.
    """
    if not isinstance(result, dict) or set(result) != {
        "loss_fn_output_type",
        "loss_fn_outputs",
        "metrics",
    }:
        return None
    metrics = result["metrics"]
    if not all(type(value) is float for value in metrics.values()):
        return None
    message = pb.ForwardBackwardOutput(
        loss_fn_output_type=result["loss_fn_output_type"], metrics=metrics
    )
    for datum in result["loss_fn_outputs"]:
        record = message.loss_fn_outputs.add(num_datums=1)
        for name, tensor in datum.items():
            if (
                set(tensor) != {"data", "dtype", "shape"}
                or tensor["dtype"] not in _DTYPES
            ):
                return None
            dtype, proto_dtype = _DTYPES[tensor["dtype"]]
            exact = np.asarray(tensor["data"])
            kinds = "f" if tensor["dtype"] == "float32" else "iu"
            if exact.ndim != 1 or (exact.size and exact.dtype.kind not in kinds):
                return None
            values = exact.astype(dtype)
            if not np.array_equal(values, exact, equal_nan=True):
                return None
            shape = tensor["shape"]
            if (
                not shape
                or prod(shape) != values.size
                or (len(shape) > 1 and 0 in shape)
            ):
                return None
            data = values.tobytes()
            record.fields[name].CopyFrom(
                pb.BatchedTensor(
                    data=data,
                    offsets=np.asarray([0, len(data)], dtype="<i8").tobytes(),
                    dtype=proto_dtype,
                    trailing_shape=shape[1:],
                )
            )
    return EncodedResult(message.SerializeToString())


def decode_forward_backward(body: bytes) -> dict[str, Any]:
    """JSON form of a result from ``encode_forward_backward_exact``."""
    message = pb.ForwardBackwardOutput.FromString(body)
    outputs = []
    for record in message.loss_fn_outputs:
        datum = {}
        for name, tensor in record.fields.items():
            dtype, label = _NUMPY_DTYPES[tensor.dtype]
            values = np.frombuffer(tensor.data, dtype=dtype)
            trailing = list(tensor.trailing_shape)
            datum[name] = {
                "data": values.tolist(),
                "dtype": label,
                "shape": [
                    values.size // prod(trailing) if trailing else values.size,
                    *trailing,
                ],
            }
        outputs.append(datum)
    return {
        "loss_fn_output_type": message.loss_fn_output_type,
        "loss_fn_outputs": outputs,
        "metrics": dict(message.metrics),
    }


def encode_result(result: dict[str, Any]) -> bytes | None:
    if "loss_fn_outputs" in result:
        message = pb.ForwardBackwardOutput(
            loss_fn_output_type=result["loss_fn_output_type"],
            metrics=result.get("metrics", {}),
        )
        # One record per datum supports different trailing shapes between rows.
        # num_datums also preserves rows whose output fields are all empty.
        for datum in result["loss_fn_outputs"]:
            record = message.loss_fn_outputs.add(num_datums=1)
            for name, tensor in datum.items():
                dtype, proto_dtype = {
                    "float32": ("<f4", pb.DTYPE_FLOAT32),
                    "int64": ("<i8", pb.DTYPE_INT64),
                }[tensor["dtype"]]
                values = np.asarray(tensor["data"], dtype=dtype)
                shape = tensor.get("shape") or [values.size]
                if tensor.get("sparse_crow_indices") is not None:
                    # BatchedTensor has no CSR representation.
                    rows = tensor["sparse_crow_indices"]
                    cols = tensor["sparse_col_indices"]
                    dense = np.zeros((len(rows) - 1, shape[-1]), dtype=dtype)
                    for row, (start, end) in enumerate(pairwise(rows)):
                        np.add.at(dense[row], cols[start:end], values[start:end])
                    values = dense
                data = values.tobytes()
                record.fields[name].CopyFrom(
                    pb.BatchedTensor(
                        data=data,
                        offsets=np.asarray([0, len(data)], dtype="<i8").tobytes(),
                        dtype=proto_dtype,
                        trailing_shape=shape[1:],
                    )
                )
        return message.SerializeToString()
    if "sequences" in result:
        message = pb.SampleResponse(
            prompt_cache_hit_tokens=result.get("prompt_cache_hit_tokens", 0),
        )
        for sequence in result["sequences"]:
            item = message.sequences.add(
                stop_reason={
                    "stop": pb.STOP_REASON_STOP,
                    "length": pb.STOP_REASON_LENGTH,
                }[sequence["stop_reason"]],
                tokens=np.asarray(sequence["tokens"], dtype="<i4").tobytes(),
            )
            if sequence.get("logprobs") is not None:
                item.logprobs = np.asarray(sequence["logprobs"], dtype="<f4").tobytes()
        if result.get("prompt_logprobs") is not None:
            message.prompt_logprobs = np.asarray(
                result["prompt_logprobs"], dtype="<f4"
            ).tobytes()
        if result.get("topk_prompt_logprobs") is not None:
            rows = result["topk_prompt_logprobs"]
            k = max((len(row or ()) for row in rows), default=0)
            tokens = np.zeros((len(rows), k), dtype="<i4")
            logprobs = np.full((len(rows), k), -99999.0, dtype="<f4")
            for i, row in enumerate(rows):
                for j, (token, logprob) in enumerate(row or ()):
                    tokens[i, j], logprobs[i, j] = token, logprob
            message.topk_prompt_logprobs.CopyFrom(
                pb.TopkPromptLogprobs(
                    token_ids=tokens.tobytes(),
                    logprobs=logprobs.tobytes(),
                    prompt_length=len(rows),
                    k=k,
                )
            )
        return message.SerializeToString()
    return None
