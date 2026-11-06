"""Preserve route values and admission checks while keeping arrays internal."""

from types import SimpleNamespace

import numpy as np
import pytest
from tinker import types

from spindle.backends.miles_runtime.replay_data import (
    add_replay_to_train_data,
    replay_row,
)


def test_array_routes_are_owned_int32_snapshots():
    source = np.array([0, 287, -1, -1], dtype=np.int64).reshape(2, 1, 2)
    tensor = types.TensorData(data=source, dtype="int64", shape=[2, 1, 2])
    row = replay_row({"routed_experts": tensor}, np.array([1, 2]))
    values = row["routed_experts"]["values"]
    assert values.dtype == np.int32
    assert values.tolist() == [0, 287, -1, -1]
    source[:] = 5
    assert values.tolist() == [0, 287, -1, -1]
    batch = {}
    add_replay_to_train_data(batch, [row])
    routes = batch["rollout_routed_experts"][0]
    assert routes.dtype == np.int32
    assert routes.tolist() == [[[0, 287]], [[-1, -1]]]
    assert np.shares_memory(routes, values)


@pytest.mark.parametrize("value", [-2, 2**31, 2**63 - 1])
def test_routes_are_range_checked_before_narrowing(value):
    tensor = types.TensorData(data=[value], dtype="int64", shape=[1, 1, 1])
    with pytest.raises(ValueError, match="int32 range"):
        replay_row({"routed_experts": tensor}, np.array([1]))


@pytest.mark.parametrize("values", [[True], [1.0], [1.5], [True, 1], [1, 1.5]])
def test_list_backed_inputs_still_reject_nonintegers(values):
    tensor = SimpleNamespace(
        data=values, dtype="int64", shape=[1, 1, len(values)], sparse_crow_indices=None
    )
    with pytest.raises(ValueError, match="dense int64"):
        replay_row({"routed_experts": tensor}, np.array([1]))


def test_empty_sampling_support_and_list_backed_routes():
    def tensor(values, shape):
        return SimpleNamespace(
            data=values, dtype="int64", shape=shape, sparse_crow_indices=None
        )

    row = replay_row(
        {
            "routed_experts": tensor([0, 1], [1, 1, 2]),
            "sampling_mask_ids": tensor([], [0]),
            "sampling_mask_offsets": tensor([0, 0], [2]),
        },
        np.array([1]),
    )
    assert row["sampling_mask_ids"] == []
    assert row["sampling_mask_offsets"] == [0, 0]
    assert row["routed_experts"]["values"].tolist() == [0, 1]
