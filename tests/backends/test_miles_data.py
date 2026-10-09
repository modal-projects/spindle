import pytest
from tinker import Datum, ModelInput, TensorData

from spindle.backends.miles_runtime.data import _datum_row


@pytest.mark.parametrize("sparse", [False, True])
def test_datum_row_owns_converted_values(sparse):
    datum = Datum(
        ModelInput.from_ints([1, 2, 3]),
        {"target_tokens": TensorData(data=[2, 3, 4], dtype="int64", shape=[3])},
    )
    datum.loss_fn_inputs["weights"] = TensorData(
        data=[0.25, 0.75] if sparse else [0.25, 0.0, 0.75],
        dtype="float32",
        shape=[3],
        sparse_crow_indices=[0, 2] if sparse else None,
        sparse_col_indices=[0, 2] if sparse else None,
    )
    original_targets = list(datum.loss_fn_inputs["target_tokens"].data)
    original_weights = list(datum.loss_fn_inputs["weights"].data)
    row = _datum_row(datum, "cross_entropy", 0)
    assert row["weights"] == [0.25, 0.0, 0.75]
    row["target_tokens"][0] = 999
    row["weights"][0] = 999
    assert datum.loss_fn_inputs["target_tokens"].data == original_targets
    assert datum.loss_fn_inputs["weights"].data == original_weights
