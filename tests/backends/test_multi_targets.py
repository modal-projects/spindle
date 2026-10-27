from types import SimpleNamespace

import pytest
import torch
from tinker import Datum, ModelInput, TensorData

from spindle.backends.contract import ForwardBatch, ForwardItem
from spindle.backends.miles_runtime.data import (
    build_outputs,
    pad_slot_rows,
    prepare_batch,
)
from spindle.backends.miles_runtime.targets import install_multi_target_hooks


def datum(targets, weights=None):
    targets = torch.tensor(targets)
    inputs = {
        "target_tokens": TensorData(
            data=targets.flatten().tolist(), dtype="int64", shape=list(targets.shape)
        )
    }
    if weights is not None:
        weights = torch.tensor(weights)
        inputs["weights"] = TensorData(
            data=weights.flatten().tolist(), dtype="float32", shape=list(weights.shape)
        )
    return Datum(model_input=ModelInput.from_ints([1, 2, 3]), loss_fn_inputs=inputs)


def test_mixed_shapes_alignment_and_outputs():
    batch = ForwardBatch(
        items=(
            ForwardItem(
                "a",
                (
                    datum([7, 8, 9]),
                    datum([[2, 4], [3, 5], [4, 6]]),
                    datum([[2], [3], [4]]),
                ),
            ),
        ),
        loss_fn="cross_entropy",
        loss_fn_config={},
    )
    prepared = prepare_batch(batch, {"a": 0}, sequence_alignment=8)
    rows = [row for _, row in prepared.slot_rows]
    assert len(rows) == 3  # Candidate labels do not duplicate model inputs.
    assert all(len(row["tokens"]) == 8 and row["target_len"] == 7 for row in rows)
    assert rows[0]["tokens"][:3] == [1, 2, 3]
    assert rows[1]["target_tokens"][:3] == [[2, 4], [3, 5], [4, 6]]
    assert rows[1]["weights"] == [[1.0, 1.0]] * 3 + [[0.0, 0.0]] * 4
    padded, _ = pad_slot_rows(prepared.slot_rows, 4)
    assert len(padded) == 4 and padded[-1][1]["weights"] == [0.0]
    raw = [
        {"loss": 1, "logprobs": [-1.0] * 7},
        {"loss": 2, "logprobs": [[-2.0, -3.0]] * 7},
        {"loss": 3, "logprobs": [[-4.0]] * 7},
    ]
    (output,) = build_outputs(batch, prepared, raw)
    assert [record["logprobs"].shape for record in output.loss_fn_outputs] == [
        [3],
        [3, 2],
        [3, 1],
    ]
    assert output.loss_fn_outputs[1]["logprobs"].data == [-2.0, -3.0] * 3
    assert output.metrics["tokens:sum"] == 9


@pytest.mark.parametrize(
    "targets,weights",
    [([[2, 4], [3, 5], [4, 6]], [1, 1, 1]), ([2, 3, 4], [[1], [1], [1]])],
)
def test_weights_must_match_target_shape(targets, weights):
    batch = ForwardBatch(
        items=(ForwardItem("a", (datum(targets, weights),)),),
        loss_fn="cross_entropy",
        loss_fn_config={},
    )
    with pytest.raises(ValueError, match="weights must have shape"):
        prepare_batch(batch, {"a": 0})


@pytest.mark.parametrize(
    "shape,data",
    [([3, 0], []), ([3, 1, 1], [2, 3, 4]), ([2, 2], [2, 3, 4, 5])],
)
def test_invalid_target_shape(shape, data):
    value = datum([2, 3, 4])
    value.loss_fn_inputs["target_tokens"] = TensorData(
        data=data, dtype="int64", shape=shape
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    with pytest.raises(ValueError, match="target_tokens"):
        prepare_batch(batch, {"a": 0})


@pytest.mark.parametrize("nested", [False, True])
def test_candidate_scores_and_gradients_share_logits(nested):
    torch.manual_seed(17)
    targets = [
        torch.tensor([2, 3, 4]),
        torch.tensor([[1, 4], [2, 5], [3, 6]]),
        torch.tensor([[2], [3], [4]]),
    ]
    if not nested:
        targets = [target[:, 0] if target.ndim == 2 else target for target in targets]
    weights = [
        torch.arange(1, target.numel() + 1).reshape(target.shape).float()
        for target in targets
    ]
    masks = [torch.tensor([1, 1, 0])] * 3
    batch = {"target_tokens": targets, "loss_masks": masks}
    logits = torch.randn(3, 3, 8, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    calls = []

    def single_target(args, column_batch, supplied_logits):
        assert supplied_logits is logits
        calls.append(supplied_logits)
        return [
            row.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
            for row, target in zip(
                supplied_logits, column_batch["target_tokens"], strict=True
            )
        ]

    losses = SimpleNamespace(
        _target_logprobs=single_target,
        _response_masks=lambda batch, _: batch["loss_masks"],
    )
    install_multi_target_hooks(losses)
    scores = losses._target_logprobs(None, batch, logits)
    expanded_masks = losses._response_masks(batch, scores)
    loss = -sum(
        (score * weight * mask).sum()
        for score, weight, mask in zip(scores, weights, expanded_masks, strict=True)
    )
    expected_loss = 0
    for row, target, weight, mask, actual in zip(
        reference_logits, targets, weights, masks, scores, strict=True
    ):
        index = target if target.ndim == 2 else target[:, None]
        expected = row.log_softmax(-1).gather(-1, index)
        if target.ndim == 1:
            expected = expected.squeeze(-1)
        else:
            mask = mask[:, None]
        torch.testing.assert_close(actual, expected)
        expected_loss -= (expected * weight * mask).sum()
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert len(calls) == (2 if nested else 1)


def test_miles_v012_masks_and_outputs_keep_candidate_columns():
    def single_target(args, batch, logits):
        return [target.float() for target in batch["target_tokens"]]

    def gather(args, batch, log_probs, per_datum_losses):
        assert all(log_prob.ndim == 1 for log_prob in log_probs)
        return [
            {"sample_index": index, "logprobs": log_prob, "loss": loss}
            for index, (log_prob, loss) in enumerate(
                zip(log_probs, per_datum_losses, strict=True)
            )
        ]

    losses = SimpleNamespace(
        _target_logprobs=single_target,
        _local_response_masks=lambda args, batch, _: batch["loss_masks"],
        _gather_per_datum_outputs=gather,
    )
    install_multi_target_hooks(losses)
    targets = [torch.tensor([1, 2]), torch.tensor([[3, 4], [5, 6]])]
    batch = {"target_tokens": targets, "loss_masks": [torch.ones(2)] * 2}
    scores = losses._target_logprobs(None, batch, None)
    masks = losses._local_response_masks(None, batch, scores)
    assert [mask.shape for mask in masks] == [(2,), (2, 1)]
    outputs = losses._gather_per_datum_outputs(None, batch, scores, [1.0, 2.0])
    torch.testing.assert_close(outputs[0]["logprobs"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        outputs[1]["logprobs"], torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    )
    assert [output["loss"] for output in outputs] == [1.0, 2.0]


def test_reported_1631_by_five_targets_use_one_prompt():
    value = Datum(
        model_input=ModelInput.from_ints([1] * 1631),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=[2, 3, 4, 5, 6] * 1631, dtype="int64", shape=[1631, 5]
            ),
        },
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    prepared = prepare_batch(batch, {"a": 0})
    assert len(prepared.slot_rows) == 1
    _, row = prepared.slot_rows[0]
    assert len(row["tokens"]) == 1632
    assert row["target_len"] == 1631
    assert row["target_tokens"][-1] == [2, 3, 4, 5, 6]


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo", "dro"])
def test_multi_target_rl_inputs_keep_candidate_dimension(loss_fn):
    value = datum([[2, 4], [3, 5], [4, 6]])
    for name in ("advantages", "logprobs"):
        value.loss_fn_inputs[name] = TensorData(
            data=[0.5] * 6, dtype="float32", shape=[3, 2]
        )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn=loss_fn, loss_fn_config={}
    )
    prepared = prepare_batch(batch, {"a": 0}, sequence_alignment=8)
    _, row = prepared.slot_rows[0]
    for name in ("advantages", "sampling_logprobs"):
        assert row[name] == [[0.5, 0.5]] * 3 + [[0.0, 0.0]] * 4


def test_sampling_support_requires_all_candidate_targets():
    value = datum([[2, 4], [3, 5], [4, 6]])
    value.loss_fn_inputs.update(
        sampling_mask_ids=TensorData(data=[4, 6], dtype="int64", shape=[2]),
        sampling_mask_offsets=TensorData(data=[0, 0, 0, 2], dtype="int64", shape=[4]),
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    assert len(prepare_batch(batch, {"a": 0}).slot_rows) == 1
    value.loss_fn_inputs["sampling_mask_ids"] = TensorData(
        data=[4, 7], dtype="int64", shape=[2]
    )
    with pytest.raises(ValueError, match="each sampling support"):
        prepare_batch(batch, {"a": 0})


def test_sdk_sparse_encoding_of_mostly_zero_2d_inputs():
    # The SDK stores 2-D target_tokens and weights as CSR when most entries are
    # zero, which is the usual shape: unweighted prompt positions.
    targets = torch.zeros(8, 2, dtype=torch.int64)
    targets[-1] = torch.tensor([4, 6])
    weights = torch.zeros(8, 2)
    weights[-1] = torch.tensor([0.5, 1.0])
    value = Datum(
        model_input=ModelInput.from_ints([1] * 8),
        loss_fn_inputs={"target_tokens": targets, "weights": weights},
    )
    assert all(
        value.loss_fn_inputs[name].sparse_crow_indices is not None
        for name in ("target_tokens", "weights")
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    _, row = prepare_batch(batch, {"a": 0}).slot_rows[0]
    assert row["target_tokens"] == targets.tolist()
    assert row["weights"] == weights.tolist()


def test_sparse_row_pointers_must_cover_values():
    value = datum([2, 3, 4])
    value.loss_fn_inputs["weights"] = TensorData(
        data=[1.0],
        dtype="float32",
        shape=[3, 2],
        sparse_crow_indices=[0, 1, 1],
        sparse_col_indices=[0],
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    with pytest.raises(ValueError, match="row pointers"):
        prepare_batch(batch, {"a": 0})


@pytest.mark.parametrize(
    "crow,columns,values",
    [
        ([0, 1, 1, 1], [2], [1.0]),
        ([0, 1, 1, 1], [-1], [1.0]),
        ([0, 2, 1, 2], [0, 1], [1.0, 2.0]),
        ([0, 1, 1, 1], [0, 1], [1.0]),
    ],
)
def test_invalid_csr_cannot_silently_move_values_between_rows(crow, columns, values):
    value = datum([[2, 4], [3, 5], [4, 6]])
    value.loss_fn_inputs["weights"] = TensorData(
        data=values,
        dtype="float32",
        shape=[3, 2],
        sparse_crow_indices=crow,
        sparse_col_indices=columns,
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    with pytest.raises(ValueError, match="CSR TensorData"):
        prepare_batch(batch, {"a": 0})


def test_csr_duplicate_columns_follow_tensor_densification():
    sparse = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 2, 3]),
        torch.tensor([1, 1, 0]),
        torch.tensor([0.25, 0.5, 1.0]),
        size=(3, 2),
    )
    value = datum([[2, 4], [3, 5], [4, 6]])
    value.loss_fn_inputs["weights"] = TensorData(
        data=sparse.values().tolist(),
        dtype="float32",
        shape=[3, 2],
        sparse_crow_indices=sparse.crow_indices().tolist(),
        sparse_col_indices=sparse.col_indices().tolist(),
    )
    batch = ForwardBatch(
        items=(ForwardItem("a", (value,)),), loss_fn="cross_entropy", loss_fn_config={}
    )
    _, row = prepare_batch(batch, {"a": 0}).slot_rows[0]
    assert row["weights"] == sparse.to_dense().tolist()
