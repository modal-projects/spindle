"""Roundtrip SDK tensors through every HTTP boundary and Miles preparation."""

from types import SimpleNamespace

import pytest
import tinker
import torch

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine import Engine
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.http import HttpEngineClient, create_engine_app
from spindle.engine.spmd import DistributedExecutor
from spindle.providers.local import InMemoryKeyValueStore
from tests.backends.test_miles import FakeMilesRuntime, _backend
from tests.support import SingleEnginePlatform, serve

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


def test_sparse_and_dense_targets_roundtrip_all_http_hops(tmp_path):
    seen = []

    class Runtime(FakeMilesRuntime):
        def forward_backward(self, slot_rows, **kwargs):
            outputs = []
            for _, row in slot_rows:
                targets = torch.tensor(row["target_tokens"])
                weights = torch.tensor(row["weights"])
                seen.append((targets, weights))
                # Distinct values expose transposition, candidate reordering,
                # and dropped zero entries in either direction of transport.
                logprobs = -targets.float() / 16
                outputs.append(
                    {
                        "logprobs": logprobs.tolist(),
                        "loss": -(weights * logprobs).sum().item(),
                    }
                )
            return outputs

    backend = _backend(tmp_path, Runtime())
    with serve(create_backend_app(DistributedExecutor(backend))) as backend_url:
        engine = Engine(HttpBackendClient(backend_url), max_models=1)
        with serve(create_engine_app(engine, token="test")) as engine_url:
            definition = SimpleNamespace(
                definition_id="targets",
                name="targets",
                model="Qwen/Qwen3-4B",
                parameterization="lora",
                max_context_length=2048,
            )
            plane = ControlPlane(
                InMemoryKeyValueStore(),
                SingleEnginePlatform(
                    "targets", HttpEngineClient(engine_url, token="test")
                ),
            )
            with serve(
                create_control_plane_app(
                    plane, (definition,), api_key="tml-test", retrieve_window=5.0
                )
            ) as url:
                service = tinker.ServiceClient(base_url=url, api_key="tml-test")
                training = service.create_lora_training_client(
                    base_model=definition.model, rank=8
                )
                targets = torch.zeros(11, 5, dtype=torch.int64)
                targets[-2:] = torch.arange(1, 11).reshape(2, 5)
                weights = torch.zeros(11, 5)
                weights[-2:] = torch.arange(1, 11).reshape(2, 5) / 16
                model_input = tinker.ModelInput.from_ints([1] * 11)
                sparse = tinker.Datum(
                    model_input, {"target_tokens": targets, "weights": weights}
                )
                assert all(
                    tensor.sparse_crow_indices is not None
                    for tensor in sparse.loss_fn_inputs.values()
                )
                dense = tinker.Datum(
                    model_input,
                    {
                        "target_tokens": tinker.TensorData(
                            data=targets.flatten().tolist(),
                            dtype="int64",
                            shape=[11, 5],
                        ),
                        "weights": tinker.TensorData(
                            data=weights.flatten().tolist(),
                            dtype="float32",
                            shape=[11, 5],
                        ),
                    },
                )
                single = tinker.Datum(
                    model_input,
                    {"target_tokens": targets[:, 0], "weights": weights[:, 0]},
                )
                for submit in (training.forward, training.forward_backward):
                    result = submit([sparse, dense, single], "cross_entropy").result(
                        timeout=30
                    )
                    expected = [
                        -targets.float() / 16,
                        -targets.float() / 16,
                        -targets[:, 0].float() / 16,
                    ]
                    for record, value in zip(
                        result.loss_fn_outputs, expected, strict=True
                    ):
                        torch.testing.assert_close(
                            record["logprobs"].to_torch(), value, atol=0, rtol=0
                        )
                    expected_loss = (
                        -(expected[0] * weights).sum() * 2
                        - (expected[2] * weights[:, 0]).sum()
                    )
                    assert result.metrics["loss:sum"] == expected_loss.item()
                assert len(seen) == 6
                for index, (received_targets, received_weights) in enumerate(seen):
                    column = index % 3 == 2
                    torch.testing.assert_close(
                        received_targets,
                        targets[:, 0] if column else targets,
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        received_weights,
                        weights[:, 0] if column else weights,
                        atol=0,
                        rtol=0,
                    )
