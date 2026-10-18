import json

import pytest
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.forward_backward_request import ForwardBackwardRequest

from spindle.engine import OperationKind
from spindle.engine.ingress import decode_forward_backward
from spindle.engine.operations import (
    parse_operation_payload,
    serialize_operation_payload,
)
from spindle.proto import tinker_public_pb2


def test_json_and_protobuf_normalize_to_the_same_forward_payload() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4]},
    )
    request = ForwardBackwardRequest(
        model_id="model-a",
        seq_id=7,
        forward_backward_input=ForwardBackwardInput(
            data=[datum],
            loss_fn="cross_entropy",
        ),
    )
    protobuf = forward_backward_request_to_proto(request).SerializeToString()
    json_body = json.dumps(
        {
            "model_id": "model-a",
            "seq_id": 7,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"tokens": [1, 2, 3]}]},
                        "loss_fn_inputs": {"target_tokens": [2, 3, 4]},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }
    ).encode()

    json_request = decode_forward_backward(json_body, "application/json")
    proto_request = decode_forward_backward(protobuf, "application/x-protobuf")

    assert (
        json_request[:3]
        == proto_request[:3]
        == (
            "model-a",
            7,
            OperationKind.FORWARD_BACKWARD,
        )
    )
    assert serialize_operation_payload(json_request[3]) == serialize_operation_payload(
        proto_request[3]
    )


def encode_forward_backward_request(**loss_fn_configs) -> bytes:
    message = tinker_public_pb2.ForwardBackwardRequest(
        model_id="model-a", seq_id=7, loss_fn="ppo", **loss_fn_configs
    )
    message.data.add().model_input.add().encoded_text.tokens = b"\x01\x00\x00\x00"
    return message.SerializeToString()


def test_protobuf_loss_fn_config_v2_numbers_are_used() -> None:
    body = encode_forward_backward_request(
        loss_fn_config={"clip_low_threshold": 0.5},
        loss_fn_config_v2={
            "clip_low_threshold": tinker_public_pb2.LossConfigValue(number=0.8),
            "clip_high_threshold": tinker_public_pb2.LossConfigValue(number=1.2),
        },
    )

    payload = decode_forward_backward(body, "application/x-protobuf")[3]

    assert payload.loss_fn_config == {
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    }


def test_protobuf_legacy_loss_fn_config_is_used_without_v2() -> None:
    body = encode_forward_backward_request(loss_fn_config={"clip_low_threshold": 0.5})

    payload = decode_forward_backward(body, "application/x-protobuf")[3]

    assert payload.loss_fn_config == {"clip_low_threshold": 0.5}


def test_protobuf_string_loss_fn_config_is_rejected() -> None:
    body = encode_forward_backward_request(
        loss_fn_config={"clip_low_threshold": 0.8},
        loss_fn_config_v2={
            "clip_low_threshold": tinker_public_pb2.LossConfigValue(number=0.8),
            "mode": tinker_public_pb2.LossConfigValue(text="token_mean"),
        },
    )

    with pytest.raises(
        ValueError, match="loss_fn_config value for mode must be a number"
    ):
        decode_forward_backward(body, "application/x-protobuf")


def test_unknown_payload_fields_are_preserved_as_extensions() -> None:
    payload = parse_operation_payload(
        OperationKind.SAVE_WEIGHTS,
        {
            "path": "checkpoint",
            "future_option": {"enabled": True},
        },
    )

    assert payload.extensions == {"future_option": {"enabled": True}}
    assert serialize_operation_payload(payload)["extensions"] == payload.extensions
