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
    padded = pad_slot_rows(prepared.slot_rows, 4)
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
