from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from tinker import types
from tinker.lib.public_interfaces.training_client import TrainingClient


@pytest.mark.parametrize("sparse", [False, True])
def test_chunk_estimate_counts_stored_values_without_materializing_data(sparse):
    tensor = types.TensorData(
        data=np.array([1.0, 2.0], dtype=np.float32),
        dtype="float32",
        shape=[100] if sparse else [2],
        sparse_crow_indices=[0, 2] if sparse else None,
        sparse_col_indices=[1, 99] if sparse else None,
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"weights": tensor},
    )
    client = SimpleNamespace(
        holder=SimpleNamespace(estimate_bytes_count_in_model_input=lambda _: 20)
    )

    def materialize(_):
        raise AssertionError("Sizing must not materialize a list")

    with patch.object(types.TensorData, "data", property(materialize)):
        assert TrainingClient._estimate_bytes_count(client, datum) == 40
