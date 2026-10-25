"""Check multi-client normalization, parallel gradients, and the Miles boundary."""

import asyncio
import math
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
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
from spindle.backends.miles_runtime.replay_data import (
    add_replay_to_train_data,
    install_bridge_replay,
)
from spindle.backends.miles_runtime.score_centering import install_losses, loss_function


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
        (ForwardItem("b", (datum(0, 0.125, -1.0),)), ForwardItem("a", (datum(),))), name
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
    request = ForwardBatch((ForwardItem("a", (datum(),)),), "score_centered_mis")
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
        (ForwardItem("a", (Datum(item.model_input, inputs),)),), "score_centered_mis"
    )
    with pytest.raises(ValueError):
        prepare_batch(request, {"a": 0})


def test_registry_and_bridge_preserve_probabilities_and_diagnostics():
    registry = SimpleNamespace(TINKER_LOSS_FUNCTIONS={"ppo": object()})
    old = registry.TINKER_LOSS_FUNCTIONS["ppo"]
    install_losses(registry, processors())
    assert registry.TINKER_LOSS_FUNCTIONS["ppo"] is old
    assert "score_centered_mis" in registry.TINKER_LOSS_FUNCTIONS

    class Bridge:
        dp_size = 1

        async def _execute_batch(self, *args):
            return "ordinary"

        async def _call_trainer(self, method, batch_id, data):
            assert data["rollout_sampling_mask_logprobs"][0].dtype == np.float32
            assert data["loss_weights"][0] == [0.25, 0.0]
            batch = batch_from_rows(rows)
            _, out = evaluate(batch, torch.zeros(1, 2, 4, requires_grad=True))
            return [out, out]  # TP replicas must not double-count diagnostics.

    runtime = SimpleNamespace(
        _build_train_data=lambda rows: {
            "loss_weights": [r["weights"] for _, r in rows]
        },
        _pad_to_dp_multiple=lambda rows, size: rows,
        ROLLOUT_DATA_VALUE_SPEC={"rollout_sampling_mask_ids": object()},
        MilesBackend=Bridge,
    )
    install_bridge_replay(runtime)
    request = ForwardBatch((ForwardItem("a", (datum(),)),), "score_centered_mis")
    rows = prepare_batch(request, {"a": 0}).slot_rows
    result = asyncio.run(
        Bridge()._execute_batch("forward_backward", 1, rows, "score_centered_mis", {})
    )
    assert len(result) == 1 and result[0]["metrics"]["tokens"] == 1
    assert (
        asyncio.run(Bridge()._execute_batch("forward_backward", 2, rows, "ppo", {}))
        == "ordinary"
    )


def distributed_check(rank, rendezvous, layout):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        cp_rank, tp_rank = divmod(rank, 2)
        request = ForwardBatch(
            (ForwardItem("a", (datum(), datum(0))),), "score_centered_mis"
        )
        batch = batch_from_rows(
            prepare_batch(request, {"a": 0}, sequence_alignment=4).slot_rows
        )
        full = torch.tensor(
            [[[2.0, 100.0, 0.0, -1.0], [0.0, 2.0, 3.0, 1.0], [3.0, 2.0, 0.0, 1.0]] * 2],
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
def test_tensor_and_context_parallel_gradients_match_reference(tmp_path, layout):
    mp.spawn(
        distributed_check, args=(f"file://{tmp_path}/rendezvous", layout), nprocs=4
    )


def test_probability_distribution_survives_miles_dp_partition_and_slot_sorting():
    from spindle.backends.miles_runtime.replay_data import install_distribution_sharding

    conversion = SimpleNamespace(
        _package_shards=lambda args, data, parts: [{"partition": p} for p in parts]
    )
    install_distribution_sharding(conversion)
    once = conversion._package_shards
    install_distribution_sharding(conversion)
    assert conversion._package_shards is once
    probabilities = [
        np.array([math.log(q)], dtype=np.float32) for q in [0.1, 0.2, 0.3, 0.4]
    ]
    shards = conversion._package_shards(
        None, {"rollout_sampling_mask_logprobs": probabilities}, [[3, 1], [2, 0]]
    )
    assert shards[0]["rollout_sampling_mask_logprobs"] == [
        probabilities[3],
        probabilities[1],
    ]
    assert shards[1]["rollout_sampling_mask_logprobs"] == [
        probabilities[2],
        probabilities[0],
    ]
