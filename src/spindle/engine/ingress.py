from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
from tinker import ModelInput
from tinker.types._pydantic_types.datum import Datum
from tinker.types._pydantic_types.tensor_data import TensorData
from tinker.types.forward_backward_input import ForwardBackwardInput

from spindle.proto import tinker_public_pb2

from .api import OperationKind
from .operations import OperationPayload, parse_operation_payload

_ENVELOPE_FIELDS = {"model_id", "seq_id", "type"}
_PROTO_DTYPES = {
    "DTYPE_FLOAT32": ("f", "float32"),
    "DTYPE_INT64": ("q", "int64"),
}


def decode_forward_backward(
    body: bytes,
    content_type: str,
) -> tuple[str, int, OperationKind, ForwardBackwardInput]:
    if content_type.startswith("application/x-protobuf"):
        message = tinker_public_pb2.ForwardBackwardRequest()
        message.ParseFromString(body)
        kind = (
            OperationKind.FORWARD
            if message.forward_only
            else OperationKind.FORWARD_BACKWARD
        )
        payload = ForwardBackwardInput(
            data=[_decode_proto_datum(item) for item in message.data],
            loss_fn=message.loss_fn,
            loss_fn_config=dict(message.loss_fn_config) or None,
        )
        model_id, seq_id = _identity(
            {"model_id": message.model_id, "seq_id": message.seq_id}, kind
        )
        return model_id, seq_id, kind, payload

    request = json.loads(body)
    nested = request.pop("forward_backward_input", None)
    if not isinstance(nested, Mapping):
        raise ValueError("forward_backward_input must be an object")
    request.update(nested)
    model_id, seq_id = _identity(request, OperationKind.FORWARD_BACKWARD)
    payload = parse_operation_payload(
        OperationKind.FORWARD_BACKWARD,
        _payload(request),
    )
    assert isinstance(payload, ForwardBackwardInput)
    return model_id, seq_id, OperationKind.FORWARD_BACKWARD, payload


def decode_json_operation(
    kind: OperationKind,
    request: Mapping[str, object],
) -> tuple[str, int, OperationPayload]:
    raw = dict(request)
    if kind == OperationKind.FORWARD:
        nested = raw.pop("forward_input", None)
        if not isinstance(nested, Mapping):
            raise ValueError("forward_input must be an object")
        raw.update(nested)
    model_id, seq_id = _identity(raw, kind)
    return model_id, seq_id, parse_operation_payload(kind, _payload(raw))


def _identity(
    request: Mapping[str, object],
    kind: OperationKind,
) -> tuple[str, int]:
    model_id = request.get("model_id")
    seq_id = request.get("seq_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{kind.value} requires model_id and seq_id")
    if isinstance(seq_id, bool) or not isinstance(seq_id, int) or seq_id <= 0:
        raise ValueError(f"{kind.value} requires model_id and seq_id")
    return model_id, seq_id


def _payload(request: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in request.items() if key not in _ENVELOPE_FIELDS}


def _decode_proto_datum(value) -> Datum:
    return Datum(
        model_input=_decode_proto_model_input(value.model_input),
        loss_fn_inputs={
            name: _decode_proto_tensor(tensor)
            for name, tensor in value.loss_fn_inputs.items()
        },
    )


def _decode_proto_model_input(value) -> ModelInput:
    chunks = []
    for chunk in value:
        kind = chunk.WhichOneof("chunk")
        if kind == "encoded_text":
            chunks.append(
                {
                    "type": "encoded_text",
                    "tokens": _decode_values(chunk.encoded_text.tokens, "i"),
                }
            )
        elif kind == "image":
            image = chunk.image
            chunks.append(
                {
                    "type": "image",
                    "data": image.data,
                    "format": image.format,
                    "expected_tokens": image.expected_tokens
                    if image.HasField("expected_tokens")
                    else None,
                }
            )
        elif kind == "dmel":
            chunks.append({"type": "dmel", "dmel": chunk.dmel.dmel})
        else:
            raise ValueError("unsupported protobuf model input chunk")
    return ModelInput.model_validate({"chunks": chunks})


def _decode_proto_tensor(value) -> TensorData:
    dtype_value = tinker_public_pb2.DType.Name(value.dtype)
    try:
        item_format, dtype = _PROTO_DTYPES[dtype_value]
    except KeyError:
        raise ValueError(f"unsupported protobuf tensor dtype: {dtype_value}") from None
    # Repeated protobuf fields have no presence bit; empty means infer shape.
    shape = list(value.shape) or None
    if value.WhichOneof("encoding") == "sparse_csr":
        sparse = value.sparse_csr
        return TensorData(
            data=_decode_values(sparse.values, item_format),
            dtype=dtype,
            shape=shape,
            sparse_crow_indices=_decode_values(sparse.crow_indices, "q"),
            sparse_col_indices=_decode_values(sparse.col_indices, "q"),
        )
    return TensorData(
        data=_decode_values(value.dense, item_format), dtype=dtype, shape=shape
    )


def _decode_values(raw: bytes, item_format: str) -> list[int | float]:
    dtype = np.dtype(f"<{item_format}")
    if len(raw) % dtype.itemsize:
        raise ValueError("protobuf tensor data has invalid byte length")
    return np.frombuffer(raw, dtype=dtype).tolist()
