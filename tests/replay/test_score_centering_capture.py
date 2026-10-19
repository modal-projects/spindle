import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest
import tinker
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto

from spindle.backends.miles_runtime.data import _datum_row
from spindle.control_plane.http import create_control_plane_app
from spindle.control_plane.service import ControlPlane
from spindle.engine.ingress import decode_forward_backward
from spindle.inference.sampling import sample_task
from spindle.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from spindle.replay import ReplaySampleResponse, capture_replay, sample_with_replay
from tests.support import TinkerStubExecutor, serve
from spindle.engine.operations import serialize_operation_payload

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


def metadata():
    return {
        "finish_reason": "length",
        "output_token_logprobs": [[math.log(0.2), 3], [math.log(0.3), 4]],
        # Sampled tokens need not be in the top-k head.
        "output_top_logprobs": [
            [[math.log(0.5), 1], [math.log(0.3), 2]],
            [[math.log(0.4), 5], [math.log(0.3), 4]],
        ],
    }


def request(**extra):
    return {
        "request_id": "sample",
        "payload": {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"max_tokens": 2, "temperature": 1.0},
            "score_centering_topk": 2,
            "num_samples": 2,
            **extra,
        },
    }


def response():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"meta_info": metadata()})

    result = asyncio.run(
        sample_task(request(), "http://sampler", transport=httpx.MockTransport(handler))
    )
    assert len(sent) == 2
    assert all(body["top_logprobs_num"] == 2 for body in sent)
    return result


@pytest.mark.parametrize("encoding", ["json", "protobuf"])
def test_capture_roundtrip_to_training_keeps_probabilities_and_positions(encoding):
    sequence = ReplaySampleResponse.model_validate(response()).sequences[0]
    assert sequence.replay.loss_fn_config() == {}
    inputs = sequence.replay.training_inputs(2)
    assert inputs["sampling_topk_token_ids"].data == [-1, -1, 1, 2, 5, 4]
    assert inputs["logprobs"].data == pytest.approx([0.0, math.log(0.2), math.log(0.3)])
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 4],
            "advantages": [0.0, 1.0, -1.0],
            **inputs,
        },
    )
    payload = types.ForwardBackwardInput(
        data=[datum],
        loss_fn="score_centering",
        loss_fn_config={"importance_sampling_clip": 2.0},
    )
    req = types.ForwardBackwardRequest(
        model_id="m", seq_id=1, forward_backward_input=payload
    )
    if encoding == "json":
        body = json.dumps(
            {
                "model_id": "m",
                "seq_id": 1,
                "forward_backward_input": serialize_operation_payload(payload),
            }
        ).encode()
        _, _, _, decoded = decode_forward_backward(body, "application/json")
    else:
        body = forward_backward_request_to_proto(req).SerializeToString()
        _, _, _, decoded = decode_forward_backward(body, "application/x-protobuf")
    row = _datum_row(decoded.data[0], decoded.loss_fn, 0)
    assert row["sampling_topk_token_ids"] == [[-1, -1], [1, 2], [5, 4]]
    assert row["sampling_topk_logprobs"][1] == pytest.approx(
        [math.log(0.5), math.log(0.3)]
    )
    assert decoded.loss_fn_config == {"importance_sampling_clip": 2.0}


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": 0.0},
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 16},
    ],
)
def test_capture_rejects_distribution_changes(params):
    task = request(sampling_params={"temperature": 1.0, **params})
    with pytest.raises(ValueError, match="capture requires"):
        asyncio.run(sample_task(task, "http://unused"))


@pytest.mark.parametrize("k", [-1, 129, 1.2, True])
def test_invalid_capture_width(k):
    with pytest.raises(ValueError, match="score_centering_topk"):
        asyncio.run(sample_task(request(score_centering_topk=k), "http://unused"))


def test_missing_topk_fails_instead_of_falling_back():
    meta = metadata()
    del meta["output_top_logprobs"]
    with pytest.raises(ValueError, match="token-aligned top-k"):
        capture_replay(
            meta,
            [3, 4],
            prompt_tokens=2,
            temperature=1.0,
            routes=False,
            mask=False,
            topk=2,
        )


def test_sdk_can_sample_and_submit_score_centering():
    seen = []

    class RecordingExecutor(TinkerStubExecutor):
        async def execute(self, model_id, kind, payload):
            seen.append(payload)
            return await super().execute(model_id, kind, payload)

    async def sampler(task):
        assert task.payload["score_centering_topk"] == 2
        return {
            "type": "sample",
            "sequences": [
                {
                    "tokens": [3, 4],
                    "logprobs": [math.log(0.2), math.log(0.3)],
                    "stop_reason": "length",
                    "replay": capture_replay(
                        metadata(),
                        [3, 4],
                        prompt_tokens=2,
                        temperature=1.0,
                        routes=False,
                        mask=False,
                        topk=2,
                    ),
                }
            ],
        }

    definition = SimpleNamespace(
        definition_id="score",
        name="score",
        model="Qwen/Qwen3-8B",
        parameterization="lora",
        max_context_length=32768,
    )
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform("score", RecordingExecutor),
        sampling_tasks=LocalSamplingTaskPlatform(sampler),
    )
    app = create_control_plane_app(
        plane, (definition,), api_key="tml-test-key", retrieve_window=5.0
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key="tml-test-key")
        sampling = service.create_sampling_client(base_model=definition.model)
        sequence = (
            sample_with_replay(
                sampling,
                types.ModelInput.from_ints([1, 2]),
                sampling_params=types.SamplingParams(max_tokens=2, temperature=1.0),
                score_centering_topk=2,
            )
            .result(timeout=30)
            .sequences[0]
        )
        training = service.create_lora_training_client(
            base_model=definition.model, rank=16
        )
        training.forward_backward(
            [
                types.Datum(
                    model_input=types.ModelInput.from_ints([1, 2, 3]),
                    loss_fn_inputs={
                        "target_tokens": [2, 3, 4],
                        "advantages": [0.0, 1.0, -1.0],
                        **sequence.replay.training_inputs(2),
                    },
                )
            ],
            loss_fn="score_centering",
        ).result(timeout=30)
    assert any(
        isinstance(payload, types.ForwardBackwardInput)
        and payload.loss_fn == "score_centering"
        for payload in seen
    )
