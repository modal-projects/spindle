import asyncio
import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tinker import Datum, ModelInput, TensorData

from spindle.backends import ForwardBatch, ForwardItem
from spindle.backends.megatron_runtime.common.forward_backward import _loss
from spindle.backends.megatron_runtime.common.score_centering import replay_logprobs
from spindle.backends.miles_runtime.data import (
    build_outputs,
    pad_slot_rows,
    prepare_batch,
)
from spindle.backends.miles_runtime.replay import install_support_transport
from spindle.backends.miles_runtime.replay_data import (
    add_replay_to_train_data,
    install_diagnostic_capture,
    replay_row,
)
from spindle.backends.miles_runtime.score_centering import install_score_centering


def request(name="score_centered_mis", *, empty=False, config=None):
    q = [0.1, 0.3, 0.6]
    datum = Datum(
        ModelInput.from_ints([1]),
        {
            "target_tokens": [0],
            "advantages": [0.0 if empty else 2.0],
            "logprobs": [0.0 if empty else math.log(q[0])],
            "sampling_mask_ids": TensorData(
                data=[] if empty else [0, 2, 3],
                dtype="int64",
                shape=[0 if empty else 3],
            ),
            "sampling_mask_offsets": TensorData(
                data=[0, 0 if empty else 3], dtype="int64", shape=[2]
            ),
            "sampling_mask_logprobs": TensorData(
                data=[] if empty else [math.log(x) for x in q],
                dtype="float32",
                shape=[0 if empty else 3],
            ),
        },
    )
    return ForwardBatch((ForwardItem("m", (datum,)),), name, config or {})


@pytest.fixture
def registry():
    def chunks(logits, **kwargs):
        start = 0
        for total, response, tokens in zip(
            kwargs["total_lengths"],
            kwargs["response_lengths"],
            kwargs["unconcat_tokens"],
            strict=True,
        ):
            yield (
                logits[0, start + total - response - 1 : start + total - 1],
                tokens[-response:],
                range(response),
            )
            start += total

    def calculate(logits, targets, group, **kwargs):
        lp = (logits / kwargs["temperature"]).log_softmax(-1)
        return lp.gather(1, targets[:, None]), -(lp.exp() * lp).sum(-1)

    def outputs(batch, logits, logprobs, per_losses):
        total = torch.stack(per_losses).sum() if per_losses else logits.sum() * 0
        return total, {
            "per_datum": [
                {"sample_index": i, "logprobs": lp.detach(), "loss": loss.detach()}
                for i, lp, loss in zip(
                    batch["sample_indices"], logprobs, per_losses, strict=True
                )
            ]
        }

    losses = SimpleNamespace(TINKER_LOSS_FUNCTIONS={}, _sum_loss_and_outputs=outputs)
    install_score_centering(
        tinker_losses=losses,
        logit_processors=SimpleNamespace(
            _iter_response_chunks=chunks, calculate_log_probs_and_entropy=calculate
        ),
        get_parallel_state=lambda: SimpleNamespace(
            tp=SimpleNamespace(rank=0, size=1, group=None), cp=SimpleNamespace(size=1)
        ),
        pad_local_shard=None,
    )
    return losses.TINKER_LOSS_FUNCTIONS


def miles_batch(req, *, alignment=1):
    prepared = prepare_batch(req, {"m": 0}, alignment)
    rows = [row for _, row in prepared.slot_rows]
    batch = {
        "unconcat_tokens": [torch.tensor(row["tokens"]) for row in rows],
        "target_tokens": [row["target_tokens"] for row in rows],
        "total_lengths": [len(row["tokens"]) for row in rows],
        "response_lengths": [row["target_len"] for row in rows],
        "rollout_log_probs": [row["sampling_logprobs"] for row in rows],
        "advantages": [row["advantages"] for row in rows],
        "loss_masks": [row["loss_mask"] for row in rows],
        "sample_indices": list(range(len(rows))),
        "loss_fn_config": req.loss_fn_config,
    }
    add_replay_to_train_data(batch, rows)
    return prepared, batch


def run(registry, name, batch, logits, *, allgather=False):
    return registry[name](
        Namespace(
            allgather_cp=allgather,
            vocab_size=4,
            rollout_temperature=3.0,
            qkv_format="thd",
            true_on_policy_mode=True,
            log_probs_chunk_size=-1,
        ),
        batch,
        logits,
        None,
    )


@pytest.mark.parametrize("name", ["mis", "score_centered", "score_centered_mis"])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_gradient_parity_and_support_replay(registry, name, temperature):
    req = request(name, config={"sampling_temperature": temperature})
    prepared, batch = miles_batch(req)
    logits = torch.tensor([[[2.0, 100.0, 0.0, -1.0], [0.0] * 4]], requires_grad=True)
    loss, outputs = run(registry, name, batch, logits)
    loss.backward()
    reference = logits.detach()[0, :1].clone().requires_grad_()
    replay = {
        "token_indices": torch.tensor([0]),
        "sampling_mask_ids": torch.tensor([0, 2, 3]),
        "sampling_mask_offsets": torch.tensor([0, 3]),
        "sampling_mask_logprobs": torch.tensor(
            batch["rollout_sampling_mask_logprobs"][0]
        ),
        "labels": torch.tensor([0]),
        "sampling_logprobs": torch.tensor(batch["rollout_log_probs"][0]),
        "advantages": torch.tensor([2.0]),
        "loss_mask": torch.ones(1),
        "loss_config": req.loss_fn_config,
        "loss_name": name,
    }
    lp, correction, _ = replay_logprobs(reference, replay)
    expected, _ = _loss(lp, {**replay, "score_correction": correction})
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logits.grad[0, :1], reference.grad)
    assert logits.grad[0, 0, 1] == 0
    if name == "mis":
        assert torch.count_nonzero(logits.grad) == 0
    elif name == "score_centered" or temperature == 1.0:
        assert torch.count_nonzero(logits.grad) > 0
    output = build_outputs(req, prepared, outputs["per_datum"])[0]
    assert output.metrics["mis_masked_fraction:mean"] == 1
    assert output.metrics["tokens:sum"] == 1
    assert output.metrics["entropy:sum"] > 0


@pytest.mark.parametrize("empty", [False, True])
def test_alignment_and_dummy_padding(registry, empty):
    req = request(empty=empty)
    prepared, batch = miles_batch(req, alignment=4)
    assert batch["response_lengths"] == [3]
    assert batch["rollout_sampling_mask_offsets"][0].tolist() == (
        [0, 0, 0, 0] if empty else [0, 3, 3, 3]
    )
    assert batch["loss_masks"][0] == ([0, 0.0, 0.0] if empty else [1, 0.0, 0.0])
    pad = pad_slot_rows(prepared.slot_rows, 2)[1][1]
    assert pad["sampling_mask_logprobs"] == []
    assert pad["loss_mask"] == [0]
    logits = torch.randn(1, 4, 4, requires_grad=True)
    loss, outputs = run(registry, req.loss_fn, batch, logits)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    if empty:
        assert loss == 0
        assert outputs["per_datum"][0]["diagnostics"]["tokens"] == 0


@pytest.mark.parametrize(
    "config",
    [
        {"mis_low": 0},
        {"mis_high": 0.1},
        {"sampling_temperature": 0},
        {"sampling_temperature": float("nan")},
        {"typo": 1},
    ],
)
def test_config_validation(config):
    with pytest.raises(ValueError):
        prepare_batch(request(config=config), {"m": 0})


@pytest.mark.parametrize(
    "field",
    [
        "advantages",
        "logprobs",
        "sampling_mask_logprobs",
        "sampling_mask_ids",
        "sampling_mask_offsets",
    ],
)
def test_required_inputs(field):
    req = request()
    del req.items[0].data[0].loss_fn_inputs[field]
    with pytest.raises(ValueError):
        prepare_batch(req, {"m": 0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.1])
def test_support_probability_validation(value):
    req = request()
    req.items[0].data[0].loss_fn_inputs["sampling_mask_logprobs"] = TensorData(
        data=[value, -1.0, -1.0], dtype="float32", shape=[3]
    )
    with pytest.raises(ValueError, match="sampling_mask_logprobs"):
        replay_row(req.items[0].data[0].loss_fn_inputs, [0])


def test_support_normalization_and_sample_agreement():
    req = request()
    req.items[0].data[0].loss_fn_inputs["logprobs"] = TensorData(
        data=[-1.0], dtype="float32", shape=[1]
    )
    with pytest.raises(ValueError, match="disagrees"):
        prepare_batch(req, {"m": 0})
    req.items[0].data[0].loss_fn_inputs["sampling_mask_logprobs"] = TensorData(
        data=[-1.0] * 3, dtype="float32", shape=[3]
    )
    with pytest.raises(ValueError, match="sum to one"):
        prepare_batch(req, {"m": 0})


def test_mis_without_support_and_allgather_rejection(registry):
    _, batch = miles_batch(request("mis"))
    for key in list(batch):
        if key.startswith("rollout_sampling_mask"):
            del batch[key]
    logits = torch.tensor([[[2.0, 100.0, 0.0, -1.0], [0.0] * 4]], requires_grad=True)
    loss, _ = run(registry, "mis", batch, logits)
    loss.backward()
    assert loss == 0 and torch.count_nonzero(logits.grad) == 0
    with pytest.raises(ValueError, match="allgather_cp"):
        run(registry, "score_centered", batch, logits, allgather=True)
    with pytest.raises(ValueError, match="require recorded"):
        run(registry, "score_centered", batch, logits)


def test_diagnostics_survive_bridge_and_idempotent_capture():
    class Backend:
        async def _call_trainer(self):
            return [{"per_datum": [{"sample_index": 0, "diagnostics": {"entropy": 3}}]}]

        async def _execute_batch(self):
            await self._call_trainer()
            return [{"loss": 0}]

    install_diagnostic_capture(Backend)
    execute = Backend._execute_batch
    install_diagnostic_capture(Backend)
    assert Backend._execute_batch is execute
    assert asyncio.run(Backend()._execute_batch()) == [
        {"loss": 0, "diagnostics": {"entropy": 3}}
    ]


def test_support_probabilities_follow_dp_partitions():
    conversion = SimpleNamespace(
        _package_shards=lambda args, data, partitions: [{} for _ in partitions]
    )
    install_support_transport(conversion)
    package = conversion._package_shards
    install_support_transport(conversion)
    assert conversion._package_shards is package
    assert package(
        None, {"rollout_sampling_mask_logprobs": [[-1], [-2], []]}, [[2, 0], [1]]
    ) == [
        {"rollout_sampling_mask_logprobs": [[], [-1]]},
        {"rollout_sampling_mask_logprobs": [[-2]]},
    ]
