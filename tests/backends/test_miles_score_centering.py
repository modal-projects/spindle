"""Check multi-client normalization, parallel gradients, and the Miles boundary."""

import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tinker import Datum, ModelInput, TensorData

from spindle.backends import ForwardBatch, ForwardItem
from spindle.backends.miles_runtime.data import (
    build_outputs,
    pad_slot_rows,
    prepare_batch,
)
from spindle.backends.miles_runtime.replay_data import add_replay_to_train_data
from spindle.backends.miles_runtime.score_centering import install_losses, loss_function
from tests.replay.test_hooks import miles as miles


def tensor(values, dtype="float32"):
    return TensorData(data=values, dtype=dtype, shape=[len(values)])


def datum(target=2, weight=0.25, advantage=2.0):
    return Datum(
        ModelInput.from_ints([1, target]),
        {
            "target_tokens": tensor([target, 3], "int64"),
            "weights": tensor([weight, 0.0]),
            "advantages": tensor([advantage, 0.0]),
            "logprobs": tensor([math.log(0.3 if target == 2 else 0.1), 0.0]),
            "sampling_mask_ids": tensor([0, 2, 3], "int64"),
            "sampling_mask_offsets": tensor([0, 3, 3], "int64"),
            "sampling_mask_logprobs": tensor([math.log(q) for q in [0.1, 0.3, 0.6]]),
        },
    )


def batch_from_rows(rows, loss="score_centered_mis"):
    datums = [row for _, row in rows]
    result = {
        "unconcat_tokens": [torch.tensor(row["tokens"]) for row in datums],
        "target_tokens": [row["target_tokens"] for row in datums],
        "loss_weights": [row["weights"] for row in datums],
        "loss_masks": [[1] * row["target_len"] for row in datums],
        "advantages": [row["advantages"] for row in datums],
        "rollout_log_probs": [row["sampling_logprobs"] for row in datums],
        "total_lengths": [len(row["tokens"]) for row in datums],
        "response_lengths": [row["target_len"] for row in datums],
        "sample_indices": list(range(len(datums))),
        "loss_fn": loss,
        "loss_fn_config": {"sampling_temperature": 0.7},
    }
    add_replay_to_train_data(result, datums)
    return result


def processors(cp=None, tp=None, indices=None):
    cp = cp or SimpleNamespace(size=1)
    tp = tp or SimpleNamespace(size=1, rank=0, group=None)

    def chunks(logits, **kwargs):
        offset = 0
        for tokens, length in zip(
            kwargs["unconcat_tokens"], kwargs["response_lengths"], strict=True
        ):
            selected = list(range(length)) if indices is None else indices(length)
            chunk = logits[0, offset : offset + len(selected)]
            offset += len(selected)
            yield chunk, tokens[torch.tensor(selected, dtype=torch.long)], selected

    return SimpleNamespace(
        get_parallel_state=lambda: SimpleNamespace(cp=cp, tp=tp),
        _iter_response_chunks=chunks,
    )


def evaluate(batch, logits, proc=None):
    return loss_function(
        Namespace(vocab_size=4, true_on_policy_mode=False),
        batch,
        logits,
        None,
        processors=proc or processors(),
    )


@pytest.mark.parametrize("name", ["mis", "score_centered", "score_centered_mis"])
def test_multi_lora_client_weights_and_masked_target_gradient(name):
    request = ForwardBatch(
        (ForwardItem("b", (datum(0, 0.125, -1.0),)), ForwardItem("a", (datum(),))),
        name,
        {"sampling_temperature": 0.7},
    )
    prepared = prepare_batch(request, {"a": 0, "b": 1}, sequence_alignment=4)
    assert prepared.locations == ((1, 0), (0, 0))
    batch = batch_from_rows(prepared.slot_rows, name)
    logits = torch.tensor([[[2.0, 100.0, 0.0, -1.0]] * 6], requires_grad=True)
    loss, outputs = evaluate(batch, logits)
    loss.backward()
    q = torch.tensor([0.1, 0.3, 0.6])
    p = torch.softmax(torch.tensor([2.0, 0.0, -1.0]) / 0.7, 0)
    ratios = p / q
    w = (
        torch.ones_like(p)
        if name == "score_centered"
        else torch.where((ratios >= 0.2) & (ratios <= 5), ratios, 0)
    )
    scores = torch.eye(3) - p[None, :]
    center = (q[:, None] * w[:, None] * scores).sum(0) if name != "mis" else 0
    for row, target, weight, advantage in [(0, 1, 0.25, 2.0), (3, 0, 0.125, -1.0)]:
        expected = -weight * advantage * (w[target] * scores[target] - center) / 0.7
        torch.testing.assert_close(logits.grad[0, row, [0, 2, 3]], expected)
    assert torch.count_nonzero(logits.grad[0, [1, 2, 4, 5]]) == 0
    assert torch.count_nonzero(logits.grad[..., 1]) == 0
    raw = [
        {**record, "logprobs": record["logprobs"].tolist()}
        for record in outputs["per_datum"]
    ]
    final = build_outputs(request, prepared, raw)
    assert len(final) == 2
    for result in final:
        assert result.metrics["tokens:sum"] == 1
        assert "entropy:sum" in result.metrics
        assert len(result.loss_fn_outputs[0]["logprobs"].data) == 2


def test_dp_padding_keeps_graph_but_contributes_no_loss_or_metrics():
    request = ForwardBatch(
        (ForwardItem("a", (datum(),)),),
        "score_centered_mis",
        {"sampling_temperature": 0.7},
    )
    rows = pad_slot_rows(prepare_batch(request, {"a": 0}).slot_rows, 2)
    batch = batch_from_rows(rows)
    logits = torch.randn(1, 3, 4, requires_grad=True)
    loss, out = evaluate(batch, logits)
    loss.backward()
    assert out["per_datum"][1]["loss"] == 0
    assert out["per_datum"][1]["metrics"]["tokens"] == 0
    assert logits.grad[0, -1].abs().sum() == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampling_mask_logprobs", None),
        ("sampling_mask_logprobs", tensor([math.log(0.2)] * 3)),
        ("sampling_mask_ids", tensor([0, 2, 2], "int64")),
    ],
)
def test_invalid_distribution_rejected_before_distributed_dispatch(field, value):
    item = datum()
    inputs = dict(item.loss_fn_inputs)
    inputs.pop(field)
    if value is not None:
        inputs[field] = value
    request = ForwardBatch(
        (ForwardItem("a", (Datum(item.model_input, inputs),)),),
        "score_centered_mis",
        {"sampling_temperature": 0.7},
    )
    with pytest.raises(ValueError):
        prepare_batch(request, {"a": 0})


def distributed_check(rank, rendezvous, layout, pre_sliced):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        cp_rank, tp_rank = divmod(rank, 2)
        request = ForwardBatch(
            (ForwardItem("a", (datum(), datum(0))),),
            "score_centered_mis",
            {"sampling_temperature": 0.7},
        )
        batch = batch_from_rows(
            prepare_batch(request, {"a": 0}, sequence_alignment=4).slot_rows
        )
        full = torch.tensor(
            [[[0.0, 100.0, 0.0, 0.0]] * 6],
            requires_grad=True,
        )
        reference, ref_out = evaluate(batch, full)
        reference.backward()

        def select(length):
            # Noncontiguous (zigzag) or contiguous CP ownership, including empty
            # support rows. Every response token belongs to exactly one rank.
            return [
                i
                for i in range(length)
                if (i % 2 if layout == "zigzag" else i >= 2) == cp_rank
            ]

        global_indices = [offset + i for offset in (0, 3) for i in select(3)]
        shard = (
            full.detach()[:, global_indices, tp_rank * 2 : tp_rank * 2 + 2]
            .clone()
            .requires_grad_()
        )
        proc = processors(
            cp=SimpleNamespace(size=2, group=cp_groups[tp_rank]),
            tp=SimpleNamespace(size=2, rank=tp_rank, group=tp_groups[cp_rank]),
            indices=select,
        )
        if pre_sliced:
            batch["rollout_log_probs"] = [
                torch.as_tensor(values)[select(len(values))]
                for values in batch["rollout_log_probs"]
            ]
        loss, out = evaluate(batch, shard, proc)
        loss.backward()
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(
            shard.grad, full.grad[:, global_indices, tp_rank * 2 : tp_rank * 2 + 2]
        )
        for actual, expected in zip(
            out["per_datum"], ref_out["per_datum"], strict=True
        ):
            torch.testing.assert_close(actual["logprobs"], expected["logprobs"])
            assert actual["metrics"] == pytest.approx(expected["metrics"])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("pre_sliced", [False, True])
def test_tensor_and_context_parallel_gradients_match_reference(
    tmp_path, layout, pre_sliced
):
    mp.spawn(
        distributed_check,
        args=(f"file://{tmp_path}/rendezvous", layout, pre_sliced),
        nprocs=4,
    )


def test_hooked_get_batch_preserves_probabilities_through_loss(miles):
    request = ForwardBatch(
        (ForwardItem("a", (datum(),)),),
        "score_centered_mis",
        {"sampling_temperature": 0.7},
    )
    data = batch_from_rows(prepare_batch(request, {"a": 0}).slot_rows)
    keys = [key for key in data if not key.startswith("rollout_sampling_mask_")]
    batch = miles.model.get_batch(SimpleNamespace(rollout_data=data), keys)
    logits = torch.tensor([[[0.0, 100.0, 0.0, 0.0]] * 2], requires_grad=True)
    loss, outputs = evaluate(batch, logits)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert outputs["per_datum"][0]["metrics"]["tokens"] == 1


@pytest.mark.parametrize("unweighted_first", [False, True])
def test_omitted_weights_mask_prompt_and_mix_with_explicit_clients(unweighted_first):
    item = datum()
    inputs = dict(item.loss_fn_inputs)
    inputs.pop("weights")
    inputs["target_tokens"] = tensor([1, 2, 3], "int64")
    inputs["advantages"] = tensor([0.0, 2.0, 0.0])
    inputs["logprobs"] = tensor([0.0, math.log(0.3), 0.0])
    inputs["sampling_mask_offsets"] = tensor([0, 0, 3, 3], "int64")
    implicit = Datum(ModelInput.from_ints([1, 1, 2]), inputs)
    rows = [implicit, datum()] if unweighted_first else [datum(), implicit]
    request = ForwardBatch(
        (ForwardItem("a", tuple(rows)),),
        "score_centered_mis",
        {"sampling_temperature": 0.7},
    )
    prepared = prepare_batch(request, {"a": 0})
    batch = batch_from_rows(prepared.slot_rows)
    index = 0 if unweighted_first else 1
    assert torch.as_tensor(batch["loss_weights"][index]).tolist() == [0.0, 1.0, 0.0]
    assert all("weights" in row for _, row in prepared.slot_rows)


@pytest.mark.parametrize("missing", ["registry", "positions", "parallel"])
def test_unsupported_miles_loss_api_fails_at_install(missing):
    def chunks(logits, *, include_response_indices):
        return iter(())

    losses = SimpleNamespace(TINKER_LOSS_FUNCTIONS={})
    proc = SimpleNamespace(
        _iter_response_chunks=chunks, get_parallel_state=lambda: None
    )
    if missing == "registry":
        del losses.TINKER_LOSS_FUNCTIONS
    elif missing == "positions":
        proc._iter_response_chunks = lambda logits: iter(())
    else:
        del proc.get_parallel_state
    with pytest.raises(RuntimeError, match="Miles exact-support replay requires"):
        install_losses(losses, proc)
