"""SDK JSON and compressed protobuf must deliver the same backend inputs."""

import asyncio
import json

import httpx
import numpy as np
import pytest
import zstandard
from tinker import types

try:
    from tinker.lib._pydantic_conv import to_pydantic_request
except ImportError:  # Tinker 0.25+ sends training requests only as protobuf.
    to_pydantic_request = None
from tinker.proto.request_conv import forward_backward_request_to_proto

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
from tests.control_plane.test_http import DEFINITION, DEFINITIONS, created_model
from tests.support import EchoExecutor


@pytest.mark.skipif(
    to_pydantic_request is None, reason="requires the Tinker 0.24 JSON request path"
)
@pytest.mark.parametrize("shape", [[4], None])
@pytest.mark.parametrize("protobuf_first", [False, True])
def test_sdk_requests_match_through_frontend_engine_and_backend(shape, protobuf_first):
    async def run():
        calls = []

        class RecordingExecutor(EchoExecutor):
            async def execute_forward_backward_batch(self, executions):
                calls.extend(executions)
                return await super().execute_forward_backward_batch(executions)

        backend = HttpBackendClient(
            "http://backend",
            transport=httpx.ASGITransport(app=create_backend_app(RecordingExecutor())),
        )
        platform = LocalEnginePlatform(DEFINITION, lambda: backend)
        plane = ControlPlane(InMemoryKeyValueStore(), platform)
        app = create_control_plane_app(plane, DEFINITIONS, retrieve_window=1.0)
        async with httpx.AsyncClient(
            base_url="http://frontend", transport=httpx.ASGITransport(app=app)
        ) as client:
            _, model_id = await created_model(client)
            request = types.ForwardBackwardRequest(
                model_id=model_id,
                seq_id=1,
                forward_backward_input=types.ForwardBackwardInput(
                    data=[
                        types.Datum(
                            model_input=types.ModelInput(
                                chunks=[
                                    types.EncodedTextChunk(tokens=[1, 2, 3, 4]),
                                    types.ImageChunk(
                                        data=b"\x89PNG\x00\xfftest",
                                        format="png",
                                        expected_tokens=16,
                                    ),
                                    types.EncodedTextChunk(tokens=[5, 6]),
                                ]
                            ),
                            loss_fn_inputs={
                                "advantages": types.TensorData(
                                    data=np.array(
                                        [0.1, -0.0, -0.2, 1e-30], dtype=np.float32
                                    ),
                                    dtype="float32",
                                    shape=shape,
                                ),
                                "target_tokens": types.TensorData(
                                    data=[1, 2, 3, 4],
                                    dtype="int64",
                                    shape=shape,
                                ),
                                "weights": types.TensorData(
                                    data=[0.5],
                                    dtype="float32",
                                    shape=[1, 4],
                                    sparse_crow_indices=[0, 1],
                                    sparse_col_indices=[2],
                                ),
                            },
                        )
                    ],
                    loss_fn="ppo",
                    loss_fn_config={"clip": 0.2},
                ),
            )
            old_json = to_pydantic_request(request).model_dump_json().encode()
            proto = forward_backward_request_to_proto(request).SerializeToString()
            expected = json.loads(old_json)["forward_backward_input"]
            # Inspect executor inputs directly, not through another tensor cast.
            requests = (
                (old_json, {"content-type": "application/json"}),
                (
                    zstandard.ZstdCompressor().compress(proto),
                    {
                        "content-type": "application/x-protobuf",
                        "content-encoding": "zstd",
                    },
                ),
            )
            if protobuf_first:
                requests = tuple(reversed(requests))
            for body, headers in requests:
                response = await client.post(
                    "/api/v1/forward_backward", content=body, headers=headers
                )
                assert response.status_code == 200, response.text
                retrieved = await client.post(
                    "/api/v1/retrieve_future",
                    json={"request_id": response.json()["request_id"]},
                )
                assert retrieved.status_code == 200, retrieved.text
            assert len(calls) == 1  # The other format is an identical retry.
            actual = calls[0].payload
            assert actual.loss_fn_config == expected["loss_fn_config"]
            for before, after in zip(expected["data"], actual.data, strict=True):
                assert (
                    after.model_input.model_dump(mode="json") == before["model_input"]
                )
                assert {
                    k: v.model_dump(mode="json")
                    for k, v in after.loss_fn_inputs.items()
                } == before["loss_fn_inputs"]
        for engine in platform._servers.values():
            await engine.close()
        await backend.close()

    asyncio.run(run())
