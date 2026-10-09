import asyncio
import json

import httpx
import numpy as np
import pytest
from tinker.proto import tinker_public_pb2 as pb

from spindle.control_plane.http import create_control_plane_app
from spindle.control_plane.service import FutureResolution, FutureResolutionStatus
from spindle.engine import Engine, FutureStatus
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.http import HttpEngineClient, create_engine_app
from spindle.engine.training_transport import decode_results, encode_results
from spindle.proto.responses import (
    EncodedResult,
    encode_forward_backward_exact,
    encode_result,
)
from tests.control_plane.test_http import DEFINITIONS
from tests.support import EchoExecutor

MODEL_SPEC = {"base_model": "test/model", "parameterization": "full"}


def forward_result(*lengths: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    for length in lengths:
        logprobs = (-rng.random(length)).astype(np.float32).tolist()
        loss = float(np.float32(-sum(logprobs)))
        records.append(
            {
                "loss:sum": {"data": [loss], "dtype": "float32", "shape": [1]},
                "logprobs": {"data": logprobs, "dtype": "float32", "shape": [length]},
            }
        )
    return {
        "loss_fn_output_type": "ArrayRecord",
        "loss_fn_outputs": records,
        "metrics": {"loss:sum": 1.25, "tokens:sum": float(sum(lengths))},
    }


def test_exact_encoding_round_trips_and_matches_frontend_encoding():
    result = forward_result(5, 0, 3)
    encoded = encode_forward_backward_exact(result)
    assert encoded is not None
    assert encoded.to_json() == result
    assert pb.ForwardBackwardOutput.FromString(encoded.body) == (
        pb.ForwardBackwardOutput.FromString(encode_result(result))
    )


def _with(result: dict, **tensor) -> dict:
    result["loss_fn_outputs"][0]["logprobs"].update(tensor)
    return result


@pytest.mark.parametrize(
    "result",
    [
        _with(forward_result(2), data=[0.1, -0.5]),
        _with(forward_result(2), data=[0, 1]),
        _with(forward_result(2), sparse_crow_indices=[0, 2]),
        _with(forward_result(2), dtype="int64", data=[1.0, 2.0]),
        _with(forward_result(2), shape=[3]),
        {**forward_result(2), "metrics": {"tokens:sum": 2}},
        {"type": "save_weights", "path": "/checkpoints/a"},
    ],
)
def test_results_that_protobuf_would_change_stay_json(result):
    assert encode_forward_backward_exact(result) is None


def test_result_frames_mix_protobuf_and_json():
    exact, other = forward_result(4), {"model_id": "model-a", "value": 0.1}
    results, telemetry = decode_results(encode_results((exact, other), {"gpu_s": 1.5}))
    assert isinstance(results[0], EncodedResult)
    assert results[0].to_json() == exact
    assert results[1] == other
    assert telemetry == {"gpu_s": 1.5}


class ForwardExecutor(EchoExecutor):
    async def execute_forward_backward_batch(self, executions):
        return tuple(
            forward_result(3, 2, seed=index) for index, _ in enumerate(executions)
        )


def _forward_body(model_id: str, seq_id: int) -> bytes:
    return json.dumps(
        {
            "model_id": model_id,
            "seq_id": seq_id,
            "forward_backward_input": {
                "data": [
                    {"model_input": {"chunks": [{"tokens": [1]}]}, "loss_fn_inputs": {}}
                ],
                "loss_fn": "cross_entropy",
            },
        }
    ).encode()


def test_engine_forwards_encoded_results_from_backend_to_frontend():
    async def run() -> None:
        backend = HttpBackendClient(
            "http://backend",
            transport=httpx.ASGITransport(app=create_backend_app(ForwardExecutor())),
        )
        engine = Engine(backend)
        await engine.accept_model("model-a", MODEL_SPEC)
        await engine.forward_backward(_forward_body("model-a", 1), "application/json")
        local = await engine.retrieve_future("model-a:1", timeout=2.0)
        assert isinstance(local.result, EncodedResult)

        client = HttpEngineClient(
            "http://engine",
            transport=httpx.ASGITransport(app=create_engine_app(engine)),
        )
        state = await client.retrieve_future("model-a:1", timeout=2.0)
        assert state.status == FutureStatus.COMPLETE
        assert state.result == local.result
        assert state.result.to_json() == forward_result(3, 2)
        await client.close()
        await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize("accept_protobuf", [False, True])
def test_frontend_serves_encoded_results_in_either_format(accept_protobuf):
    result = forward_result(6, 1)
    encoded = encode_forward_backward_exact(result)

    class Plane:
        async def retrieve(self, request_id, timeout):
            return FutureResolution(
                request_id, FutureResolutionStatus.COMPLETE, encoded
            )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://frontend",
            transport=httpx.ASGITransport(
                app=create_control_plane_app(Plane(), DEFINITIONS)
            ),
        ) as client:
            response = await client.post(
                "/api/v1/retrieve_future",
                json={"request_id": "a:1"},
                headers={"Accept": "application/x-protobuf"} if accept_protobuf else {},
            )
        assert response.status_code == 200
        if accept_protobuf:
            assert response.headers["content-type"] == "application/x-protobuf"
            assert response.content == encoded.body
        else:
            assert response.json() == result

    asyncio.run(run())
