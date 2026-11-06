"""Array admission preserves data ownership and rejects invalid replay data."""

import numpy as np
import pytest
from tinker import types

from spindle.backends.miles_runtime.replay_data import (
    add_replay_to_train_data,
    replay_row,
)
from spindle.backends.score_centering import validate_support
from tests.backends.test_miles_score_centering import datum, tensor


def test_replay_admission_owns_routes_and_sampling_arrays():
    item = datum()
    inputs = dict(item.loss_fn_inputs)
    inputs["routed_experts"] = types.TensorData(
        data=[0, 287, -1, -1], dtype="int64", shape=[2, 1, 2]
    )
    row = replay_row(inputs, np.array([2, 3]))
    saved = {
        name: values.copy() for name, values in row.items() if name != "routed_experts"
    }
    for value in inputs.values():
        value.to_numpy()[:] = 0
    assert row["routed_experts"]["values"].dtype == np.int32
    assert row["routed_experts"]["values"].tolist() == [0, 287, -1, -1]
    for name, values in saved.items():
        np.testing.assert_array_equal(row[name], values)
    row["target_len"] = 2
    data = {}
    add_replay_to_train_data(data, [row])
    assert np.shares_memory(
        data["rollout_routed_experts"][0], row["routed_experts"]["values"]
    )


@pytest.mark.parametrize("value", [-2, 2**31, 2**63 - 1])
def test_routes_are_checked_before_int32_conversion(value):
    route = types.TensorData(data=[value], dtype="int64", shape=[1, 1, 1])
    with pytest.raises(ValueError, match="int32 range"):
        replay_row({"routed_experts": route}, np.array([1]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampling_mask_ids", tensor([0, 2, 2], "int64")),
        ("sampling_mask_ids", tensor([0, -2, 3], "int64")),
        ("sampling_mask_ids", tensor([0, 1, 3], "int64")),
        ("sampling_mask_offsets", tensor([0, 3, 2], "int64")),
        ("sampling_mask_offsets", tensor([0, 4, 3], "int64")),
        ("sampling_mask_offsets", tensor([0, 0, 3], "int64")),
        ("sampling_mask_logprobs", tensor([float("nan"), -1.0, -1.0])),
        ("sampling_mask_logprobs", tensor([float("inf"), -1.0, -1.0])),
        ("sampling_mask_logprobs", tensor([0.1, -1.0, -1.0])),
        ("sampling_mask_logprobs", tensor([-1.0, -1.0, -1.0])),
        ("logprobs", tensor([-3.0, 0.0])),
    ],
)
def test_ragged_support_rejects_invalid_data(field, value):
    inputs = dict(datum().loss_fn_inputs)
    inputs[field] = value
    with pytest.raises(ValueError):
        validate_support(inputs, [2, 3], [0.25, 0.0], centered=True)


def test_support_allows_unsorted_ids_and_unused_empty_rows():
    inputs = dict(datum().loss_fn_inputs)
    inputs["sampling_mask_ids"] = tensor([3, 0, 2], "int64")
    inputs["sampling_mask_logprobs"] = tensor(np.log([0.6, 0.1, 0.3]).tolist())
    support = validate_support(inputs, [2, 3], [0.25, 0.0], centered=True)
    assert support["sampling_mask_ids"].tolist() == [3, 0, 2]
    assert support["sampling_mask_offsets"].tolist() == [0, 3, 3]


@pytest.mark.parametrize("width", [1, 4, 8, 9, 512])
def test_support_validation_preserves_small_and_wide_distributions(width):
    ids = np.random.default_rng(0).permutation(width)
    probabilities = np.arange(1, width + 1, dtype=np.float64)
    probabilities /= probabilities.sum()
    logq = np.log(probabilities).astype(np.float32)
    inputs = {
        "sampling_mask_ids": tensor(ids.tolist(), "int64"),
        "sampling_mask_offsets": tensor([0, 0, width], "int64"),
        "sampling_mask_logprobs": tensor(logq.tolist()),
        "logprobs": tensor([0.0, float(logq[0])]),
    }
    support = validate_support(inputs, [0, int(ids[0])], [0.0, 1.0], centered=True)
    np.testing.assert_array_equal(support["sampling_mask_ids"].numpy(), ids)
    np.testing.assert_array_equal(support["sampling_mask_logprobs"].numpy(), logq)
    if width > 1:
        duplicates = ids.copy()
        duplicates[-1] = duplicates[0]
        inputs["sampling_mask_ids"] = tensor(duplicates.tolist(), "int64")
        with pytest.raises(ValueError, match="duplicate"):
            validate_support(inputs, [0, int(ids[0])], [0.0, 1.0], centered=True)
    inputs["sampling_mask_ids"] = tensor((ids + width).tolist(), "int64")
    with pytest.raises(ValueError, match="recorded support"):
        validate_support(inputs, [0, int(ids[0])], [0.0, 1.0], centered=True)
