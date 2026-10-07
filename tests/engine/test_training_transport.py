import asyncio
import json
import struct

import numpy as np
import pytest
from tinker import types

try:
    from tinker.lib._pydantic_conv import to_pydantic_request
except ImportError:  # Tinker 0.25+ sends training requests only as protobuf.
    to_pydantic_request = None
from tinker.proto.request_conv import forward_backward_request_to_proto

from spindle.engine import Engine
from spindle.engine.api import Command, OperationKind
from spindle.engine.ingress import decode_forward_backward
from spindle.engine.operations import (
    parse_operation_payload,
    serialize_operation_payload,
)
from spindle.engine.training_transport import (
    decode_batch,
    decode_payload,
    encode_batch,
    encode_payload,
)
from spindle.errors import SequenceConflict
from tests.support import EchoExecutor


def payload():
    return parse_operation_payload(
        OperationKind.FORWARD_BACKWARD,
        {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [0, 1, 2147483647]}]},
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": [1, 2147483647, 2],
                            "dtype": "int64",
                            "shape": [3],
                        },
                        "advantages": {
                            "data": [0.123456789012345, -0.0, -1e-30],
                            "dtype": "float32",
                            "shape": [3],
                        },
                        "routed_experts": {
                            "data": [-1, 2, 3],
                            "dtype": "int64",
                            "shape": [3, 1, 1],
                        },
                        "weights": {
                            "data": [0.5],
                            "dtype": "float32",
                            "shape": [3],
                            "sparse_crow_indices": [0, 1],
                            "sparse_col_indices": [2],
                        },
                    },
                }
            ],
            "loss_fn": "ppo",
            "loss_fn_config": {"clip": 0.2},
        },
    )


def test_binary_preserves_exact_values_shapes_sparse_fields_and_metadata():
    original = payload()
    result = decode_payload(encode_payload(original))
    assert serialize_operation_payload(result) == serialize_operation_payload(original)
    # Equality alone misses negative zero and loses precision under approximations.
    before = original.data[0].loss_fn_inputs["advantages"].data
    after = result.data[0].loss_fn_inputs["advantages"].data
    assert struct.pack("<3d", *before) == struct.pack("<3d", *after)
    commands = (
        Command("a", OperationKind.FORWARD_BACKWARD, original),
        Command("b", OperationKind.FORWARD_BACKWARD, original),
    )
    decoded = decode_batch(encode_batch(commands))
    assert [item.model_id for item in decoded] == ["a", "b"]
    assert all(
        serialize_operation_payload(item.payload)
        == serialize_operation_payload(original)
        for item in decoded
    )


def test_encoding_is_independent_of_map_order_and_reuses_admitted_bytes():
    original = payload()
    admitted = encode_payload(original)
    datum = original.data[0]
    reordered = dict(reversed(list(datum.loss_fn_inputs.items())))
    datum.loss_fn_inputs.clear()
    datum.loss_fn_inputs.update(reordered)
    assert encode_payload(original) == admitted
    command = Command("a", OperationKind.FORWARD_BACKWARD, original, admitted)
    # Payload ownership stays with the engine. Prove dispatch uses the admitted
    # representation rather than accidentally doing another full conversion.
    datum.loss_fn_inputs["advantages"].data[0] = 42
    result = decode_batch(encode_batch((command,)))[0].payload
    assert result.data[0].loss_fn_inputs["advantages"].data[0] != 42


@pytest.mark.parametrize(
    "body",
    [b"", b"SPND1\0\x01", b"SPND1\0" + struct.pack("<Q", 100) + b"bad", b"pickle"],
)
def test_malformed_frames_are_rejected(body):
    with pytest.raises((ValueError, IndexError)):
        decode_payload(body)


@pytest.mark.skipif(
    to_pydantic_request is None, reason="requires the Tinker 0.24 JSON request path"
)
def test_json_and_protobuf_retries_share_identity_but_changed_data_conflicts():
    async def run():
        engine = Engine(EchoExecutor())
        await engine.accept_model(
            "a", {"base_model": "test/model", "parameterization": "full"}
        )
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2]),
            loss_fn_inputs={
                "target_tokens": [2, 3],
                "advantages": np.array([0.1, -0.2], dtype=np.float32),
            },
        )
        request = types.ForwardBackwardRequest(
            model_id="a",
            seq_id=1,
            forward_backward_input=types.ForwardBackwardInput(
                data=[datum], loss_fn="ppo"
            ),
        )
        json_body = to_pydantic_request(request).model_dump_json().encode()
        proto_body = forward_backward_request_to_proto(request).SerializeToString()
        first = await engine.forward_backward(json_body, "application/json")
        assert (
            await engine.forward_backward(proto_body, "application/x-protobuf") == first
        )
        changed = json.loads(json_body)
        changed["forward_backward_input"]["data"][0]["model_input"]["chunks"][0][
            "tokens"
        ][0] = 7
        with pytest.raises(SequenceConflict):
            await engine.forward_backward(
                json.dumps(changed).encode(), "application/json"
            )
        await engine.close()

    asyncio.run(run())


def test_multimodal_chunks_survive_protobuf_and_binary_transport():
    data = b"\x89PNG\x00\xfftest"
    request = types.ForwardBackwardRequest(
        model_id="images",
        seq_id=1,
        forward_backward_input=types.ForwardBackwardInput(
            data=[
                types.Datum(
                    model_input=types.ModelInput(
                        chunks=[
                            types.EncodedTextChunk(tokens=[1, 2]),
                            types.ImageChunk(
                                data=data, format="png", expected_tokens=16
                            ),
                        ]
                    ),
                    loss_fn_inputs={},
                )
            ],
            loss_fn="cross_entropy",
        ),
    )
    body = forward_backward_request_to_proto(request).SerializeToString()
    payload = decode_forward_backward(body, "application/x-protobuf")[3]
    result = decode_payload(encode_payload(payload))
    assert result.data[0].model_input.chunks[1].data == data
    assert serialize_operation_payload(result) == serialize_operation_payload(payload)


@pytest.mark.parametrize(
    "dtype,values", [("int64", [1.5, -0.0]), ("float32", [2**53 + 1, -(2**53) - 1])]
)
def test_transport_does_not_cast_values_based_only_on_declared_dtype(dtype, values):
    original = parse_operation_payload(
        OperationKind.FORWARD_BACKWARD,
        {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 2]}]},
                    "loss_fn_inputs": {
                        "weights": {"data": values, "dtype": dtype, "shape": [2]}
                    },
                }
            ],
            "loss_fn": "cross_entropy",
        },
    )
    result = decode_payload(encode_payload(original))
    assert serialize_operation_payload(result) == serialize_operation_payload(original)
    assert result.data[0].loss_fn_inputs["weights"].data == values


@pytest.mark.parametrize("seed", range(12))
def test_binary_matches_previous_json_backend_path(seed):
    """Compare executor inputs against the old serialize/JSON/parse sequence."""
    rng = np.random.default_rng(seed)
    raw = serialize_operation_payload(payload())
    raw["data"] = [
        {
            "model_input": {
                "chunks": [{"tokens": rng.integers(0, 150000, 37).tolist()}]
            },
            "loss_fn_inputs": {
                "advantages": {
                    "data": rng.normal(size=37).tolist(),
                    "dtype": "float32",
                    "shape": [37] if seed % 2 else None,
                },
                "target_tokens": {
                    "data": [-(2**63), 2**63 - 1, 2**53 + 1],
                    "dtype": "int64",
                    "shape": [3],
                },
                "weights": {
                    "data": [-0.0, 5e-324, 1.7976931348623157e308],
                    "dtype": "float32",
                    "shape": [3],
                },
            },
        }
        for _ in range(3)
    ] + raw["data"]
    original = parse_operation_payload(OperationKind.FORWARD_BACKWARD, raw)
    previous = parse_operation_payload(
        OperationKind.FORWARD_BACKWARD,
        json.loads(json.dumps(serialize_operation_payload(original))),
    )
    actual = decode_payload(encode_payload(original))
    assert actual.loss_fn == previous.loss_fn
    assert actual.loss_fn_config == previous.loss_fn_config
    assert len(actual.data) == len(previous.data)
    for before, after in zip(previous.data, actual.data, strict=True):
        assert before.model_input == after.model_input
        assert before.loss_fn_inputs.keys() == after.loss_fn_inputs.keys()
        for name, tensor in before.loss_fn_inputs.items():
            result = after.loss_fn_inputs[name]
            assert tensor.dtype == result.dtype
            assert tensor.shape == result.shape
            assert tensor.sparse_crow_indices == result.sparse_crow_indices
            assert tensor.sparse_col_indices == result.sparse_col_indices
            assert len(tensor.data) == len(result.data)
            for x, y in zip(tensor.data, result.data, strict=True):
                assert type(x) is type(y)
                if isinstance(x, float):
                    assert struct.pack("<d", x) == struct.pack("<d", y)
                else:
                    assert x == y
