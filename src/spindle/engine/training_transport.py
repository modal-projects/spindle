"""Binary arrays for the private engine/backend connection.

Public requests are validated at engine admission. This format preserves their
values (including float64 Python values in a float32-labelled tensor) and carries
only small metadata as JSON. It never unpickles executable objects.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable

import numpy as np
from tinker import ModelInput
from tinker.types._pydantic_types.datum import Datum
from tinker.types._pydantic_types.tensor_data import TensorData
from tinker.types.encoded_text_chunk import EncodedTextChunk
from tinker.types.forward_backward_input import ForwardBackwardInput

from spindle.encoding import canonical_json

from .api import Command, OperationKind

CONTENT_TYPE = "application/vnd.spindle.training.v1"
_MAGIC = b"SPND1\0"
_SIZE = struct.Struct("<Q")


def _bundle(parts: Iterable[bytes]) -> bytes:
    framed = [_MAGIC]
    for part in parts:
        framed.extend((_SIZE.pack(len(part)), part))
    return b"".join(framed)


def _parts(body: bytes | memoryview) -> list[memoryview]:
    if body[: len(_MAGIC)] != _MAGIC:
        raise ValueError("unsupported training transport version")
    view = memoryview(body)
    parts = []
    offset = len(_MAGIC)
    while offset < len(view):
        if len(view) - offset < _SIZE.size:
            raise ValueError("truncated training frame")
        size = _SIZE.unpack_from(view, offset)[0]
        offset += _SIZE.size
        if size > len(view) - offset:
            raise ValueError("truncated training buffer")
        parts.append(view[offset : offset + size])
        offset += size
    return parts


def encode_payload(payload: ForwardBackwardInput) -> bytes:
    buffers: list[bytes] = []

    def array(values, dtype=None):
        # The public schema labels a tensor's intended dtype but accepts both
        # integer and float lists. Preserve received values; do not silently
        # truncate floats or round large integers during transport.
        if dtype is None:
            packed = np.asarray(values)
            if packed.dtype.kind in "iu":
                if (
                    packed.dtype.kind == "u"
                    and packed.size
                    and packed.max() > np.iinfo(np.int64).max
                ):
                    raise ValueError("training integer exceeds int64")
                dtype = "<i8"
            elif packed.dtype.kind == "f":
                dtype = "<f8"
            else:
                raise ValueError("training tensor must contain numeric values")
            packed = packed.astype(dtype, copy=False)
        else:
            packed = np.asarray(values, dtype=dtype)
        if packed.ndim != 1:
            raise ValueError("training tensor data must be flat")
        if packed.dtype.kind == "f" and not np.isfinite(packed).all():
            raise ValueError("training tensor contains non-finite values")
        index = len(buffers) + 1
        buffers.append(packed.tobytes())
        return {"buffer": index, "dtype": dtype}

    data = []
    for datum in payload.data:
        chunks = []
        for chunk in datum.model_input.chunks:
            if chunk.type == "encoded_text":
                # Exclude tokens before dumping: no traversal of the token array.
                value = chunk.model_dump(
                    mode="json", exclude={"tokens"}, exclude_defaults=True
                )
                value["tokens"] = array(chunk.tokens, "<i8")
            else:
                value = chunk.model_dump(mode="json", exclude_defaults=True)
            chunks.append(value)
        inputs = {}
        for name, tensor in sorted(datum.loss_fn_inputs.items()):
            if tensor.dtype not in {"int64", "float32"}:
                raise ValueError(f"unsupported training dtype: {tensor.dtype}")
            inputs[name] = {
                "data": array(tensor.data),
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "sparse_crow_indices": None
                if tensor.sparse_crow_indices is None
                else array(tensor.sparse_crow_indices, "<i8"),
                "sparse_col_indices": None
                if tensor.sparse_col_indices is None
                else array(tensor.sparse_col_indices, "<i8"),
            }
        data.append({"chunks": chunks, "inputs": inputs})
    metadata = canonical_json(
        {
            "data": data,
            "loss_fn": payload.loss_fn,
            "loss_fn_config": payload.loss_fn_config,
        }
    ).encode()
    return _bundle((metadata, *buffers))


def decode_payload(body: bytes | memoryview) -> ForwardBackwardInput:
    parts = _parts(body)
    if not parts:
        raise ValueError("missing training metadata")
    metadata = json.loads(bytes(parts[0]))

    def array(value):
        if value is None:
            return None
        index, dtype = value["buffer"], value["dtype"]
        if (
            not isinstance(index, int)
            or not 0 < index < len(parts)
            or dtype not in {"<i8", "<f8"}
        ):
            raise ValueError("invalid training buffer reference")
        # Lists match the validated Pydantic tensor representation used by the
        # existing backend. No text-number parsing or second schema walk.
        return np.frombuffer(parts[index], dtype=dtype).tolist()

    data = []
    for value in metadata["data"]:
        chunks = []
        for chunk in value["chunks"]:
            if "tokens" in chunk:
                chunks.append(
                    EncodedTextChunk.model_construct(
                        **{**chunk, "tokens": array(chunk["tokens"])}
                    )
                )
            else:
                chunks.extend(ModelInput.model_validate({"chunks": [chunk]}).chunks)
        model_input = ModelInput.model_construct(chunks=chunks)
        inputs = {
            name: TensorData.model_construct(
                data=array(tensor["data"]),
                dtype=tensor["dtype"],
                shape=tensor["shape"],
                sparse_crow_indices=array(tensor["sparse_crow_indices"]),
                sparse_col_indices=array(tensor["sparse_col_indices"]),
            )
            for name, tensor in value["inputs"].items()
        }
        data.append(
            Datum.model_construct(model_input=model_input, loss_fn_inputs=inputs)
        )
    return ForwardBackwardInput(
        data=data,
        loss_fn=metadata["loss_fn"],
        loss_fn_config=metadata["loss_fn_config"],
    )


def encode_batch(commands: tuple[Command, ...]) -> bytes:
    parts = []
    for command in commands:
        if command.kind != OperationKind.FORWARD_BACKWARD or not isinstance(
            command.payload, ForwardBackwardInput
        ):
            raise ValueError("training batch requires forward_backward commands")
        parts.extend(
            (
                command.model_id.encode(),
                command.encoded_payload or encode_payload(command.payload),
            )
        )
    return _bundle(parts)


def decode_batch(body: bytes) -> tuple[Command, ...]:
    parts = _parts(body)
    if not parts or len(parts) % 2:
        raise ValueError("invalid training batch")
    return tuple(
        Command(
            bytes(parts[i]).decode(),
            OperationKind.FORWARD_BACKWARD,
            decode_payload(parts[i + 1]),
        )
        for i in range(0, len(parts), 2)
    )
