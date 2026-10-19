"""Independent gradient oracles for the score-centering correction."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from tinker import Datum, ModelInput, TensorData

torch = pytest.importorskip("torch")
from spindle.backends import ForwardBatch, ForwardItem  # noqa: E402
from spindle.backends.score_centering import score_centering_loss, selected_logprobs  # noqa: E402
from spindle.backends.megatron_runtime.common.forward_backward import (  # noqa: E402
    _loss,
    build_sequence_batches,
    pack_microbatches,
    shard_microbatches,
)
from spindle.backends.miles_runtime.data import prepare_batch, pad_slot_rows  # noqa: E402
from spindle.backends.miles_runtime.score_centering import compute_score_centering_loss  # noqa: E402
from spindle.score_centering import score_centering_config  # noqa: E402


def datum(k=2, advantage=1.0):
    return Datum(
        model_input=ModelInput.from_ints([0, 1, 2]),
        loss_fn_inputs={
            "target_tokens": [1, 2, 3],
            "logprobs": [0.0, math.log(0.3), math.log(0.2)],
            "advantages": [0.0, advantage, -advantage],
            "sampling_topk_token_ids": TensorData(
                data=[-1] * k + list(range(k)) * 2, dtype="int64", shape=[3, k]
            ),
            "sampling_topk_logprobs": TensorData(
                data=[0.0] * k + [math.log(0.4)] * k * 2, dtype="float32", shape=[3, k]
            ),
        },
    )


@pytest.mark.parametrize("clip", [None, 1.2])
def test_topk_gradient_matches_full_vocabulary_tail_model(clip):
    logits = torch.tensor([[0.2, -0.8, 0.5, -1.2]], requires_grad=True)
    p = logits.softmax(-1).detach()
    head = torch.tensor([[0, 2]])
    q_head = torch.tensor([[0.15, 0.6]])
    rho = (1 - q_head.sum()) / (1 - p[:, [0, 2]].sum())
    q = p * rho
    q[:, [0, 2]] = q_head
    advantage = torch.tensor([-2.0])
    config = {} if clip is None else {"importance_sampling_clip": clip}
    # The sampled token is outside the top-k head.
    actual_lp = selected_logprobs(logits, torch.tensor([[1, 0, 2]]))
    loss = score_centering_loss(
        actual_lp[:, 0],
        actual_lp[:, 1:],
        q[:, 1].log(),
        q_head.log(),
        head >= 0,
        advantage,
        config,
    ).sum()
    loss.backward()
    actual_grad = logits.grad.clone()
    logits.grad = None
    lp = logits.log_softmax(-1)
    weight = torch.ones_like(q) if clip is None else (p / q).clamp_max(clip)
    expected = -advantage * (weight[:, 1] * lp[:, 1] - (q * weight * lp).sum(-1))
    expected.sum().backward()
    torch.testing.assert_close(actual_grad, logits.grad, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("clip", [None, 1.2])
def test_constant_reward_has_zero_expected_gradient_under_mismatch(clip):
    logits = torch.tensor([[0.2, -0.8, 0.5, -1.2]], requires_grad=True)
    q = torch.tensor([[0.1, 0.4, 0.2, 0.3]])
    lp = logits.log_softmax(-1)
    config = {} if clip is None else {"importance_sampling_clip": clip}
    expected_loss = sum(
        q[0, y]
        * score_centering_loss(
            lp[:, y],
            lp,
            q[:, y].log(),
            q.log(),
            torch.ones_like(q, dtype=torch.bool),
            torch.ones(1),
            config,
        ).sum()
        for y in range(4)
    )
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)


def test_on_policy_gradient_is_reinforce_and_padding_has_no_gradient():
    logits = torch.tensor([[0.2, -0.8, 0.5], [10.0, -10.0, -5.0]], requires_grad=True)
    lp = selected_logprobs(logits, torch.tensor([[2, 0, 1], [1, -1, -1]]), chunk_size=1)
    q = logits.softmax(-1).detach()
    loss = score_centering_loss(
        lp[:, 0],
        lp[:, 1:],
        q[:, 2].log(),
        q[:, :2].log(),
        torch.tensor([[True, True], [False, False]]),
        torch.tensor([2.0, 0.0]),
        {},
    ).sum()
    loss.backward()
    expected = 2 * q
    expected[0, 2] -= 2
    expected[1] = 0
    torch.testing.assert_close(logits.grad, expected)


def test_selected_logprobs_and_gradient_ignore_padded_vocabulary():
    logits = torch.tensor(
        [[2.0, 1.0, -1.0, 100.0], [1.0, 2.0, 3.0, 100.0]], requires_grad=True
    )
    ids = torch.tensor([[0, 2, -1], [2, 2, 1]])
    result = selected_logprobs(logits, ids, vocab_size=3, chunk_size=1)
    weights = torch.tensor([[1.0, -3.0, 8.0], [2.0, -1.0, 4.0]])
    (result * weights).sum().backward()
    reference = logits.detach().clone().requires_grad_()
    full = (
        reference[:, :3]
        .log_softmax(-1)
        .gather(1, ids.clamp_min(0))
        .masked_fill(ids < 0, 0)
    )
    (full * weights).sum().backward()
    torch.testing.assert_close(result, full)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_saturated_probabilities_have_finite_loss_and_gradient():
    logits = torch.tensor([[100.0, -100.0, -100.0]], requires_grad=True)
    lp = selected_logprobs(logits, torch.tensor([[1, 0, 2]]))
    loss = score_centering_loss(
        lp[:, 0],
        lp[:, 1:],
        torch.tensor([-300.0]),
        torch.tensor([[0.0, -300.0]]),
        torch.tensor([[True, True]]),
        torch.ones(1),
        {"importance_sampling_clip": 2.0},
    ).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_packing_mixed_head_widths_and_dp_padding_preserve_gradients():
    request = ForwardBatch(
        items=(ForwardItem("a", (datum(1),)), ForwardItem("b", (datum(2, -2),))),
        loss_fn="score_centering",
    )
    sequences = build_sequence_batches(
        request, {"a": 0, "b": 1}, max_slots=2, max_seq_length=8
    )
    packed = pack_microbatches(
        sequences, max_tokens=16, pad_to_multiple=4, total_pad_to_multiple=16
    )[0]
    assert packed["sampling_topk_token_ids"].shape == (16, 2)
    torch.manual_seed(8)
    logits = torch.randn(16, 4, requires_grad=True)

    def evaluate(batch, x):
        ids = torch.cat(
            [batch["labels"].unsqueeze(-1), batch["sampling_topk_token_ids"]], -1
        )
        lp = selected_logprobs(x, ids)
        return _loss(lp[:, 0], batch, lp[:, 1:])[0]

    packed_loss = evaluate(packed, logits)
    expected = sum(
        evaluate(sequence, logits[start : start + 3])
        for sequence, start in zip(sequences, packed["starts"], strict=True)
    )
    torch.testing.assert_close(packed_loss, expected)
    packed_grad = torch.autograd.grad(packed_loss, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(packed_grad, reference_grad)
    dummy = shard_microbatches([packed], data_parallel_rank=1, data_parallel_size=2)[0]
    assert evaluate(dummy, logits).item() == 0


def test_miles_alignment_and_loss_match_megatron():
    request = ForwardBatch(
        items=(ForwardItem("a", (datum(),)),), loss_fn="score_centering"
    )
    prepared = prepare_batch(request, {"a": 0}, sequence_alignment=8)
    row = prepared.slot_rows[0][1]
    assert np.asarray(row["sampling_topk_token_ids"]).shape == (7, 2)
    assert row["sampling_topk_token_ids"][3:] == [[-1, -1]] * 4
    padded = pad_slot_rows(prepared.slot_rows, 2)[1][1]
    assert padded["advantages"] == [0.0]
    assert padded["sampling_topk_token_ids"] == [[-1, -1]]
    batch = {
        key: [row[key]]
        for key in (
            "target_tokens",
            "advantages",
            "sampling_topk_token_ids",
            "sampling_topk_logprobs",
        )
    }
    batch.update(
        unconcat_tokens=[torch.tensor(row["tokens"])],
        total_lengths=[8],
        response_lengths=[7],
        rollout_log_probs=[row["sampling_logprobs"]],
        loss_masks=[[1] * 7],
        loss_fn_config={},
    )

    def chunks(logits, **kwargs):
        assert kwargs["include_response_indices"]
        assert kwargs["args"].rollout_temperature == 1
        yield logits[0, :7], None, range(7)

    parallel = SimpleNamespace(
        tp=SimpleNamespace(group=None, rank=0, size=1), cp=SimpleNamespace(size=1)
    )
    torch.manual_seed(2)
    logits = torch.randn(1, 8, 4, requires_grad=True)
    outputs, losses = compute_score_centering_loss(
        SimpleNamespace(rollout_temperature=0.7, vocab_size=4),
        batch,
        logits,
        logit_processors=SimpleNamespace(_iter_response_chunks=chunks),
        parallel=parallel,
    )
    megatron = build_sequence_batches(request, None, max_slots=None, max_seq_length=8)[
        0
    ]
    ids = torch.cat(
        [megatron["labels"].unsqueeze(-1), megatron["sampling_topk_token_ids"]], -1
    )
    lp = selected_logprobs(logits[0, :3], ids)
    reference = _loss(lp[:, 0], megatron, lp[:, 1:])[0]
    torch.testing.assert_close(losses[0], reference)
    torch.testing.assert_close(outputs[0][:3], lp[:, 0])
    torch.testing.assert_close(
        torch.autograd.grad(losses[0], logits, retain_graph=True)[0],
        torch.autograd.grad(reference, logits)[0],
    )


@pytest.mark.parametrize("backend", ["miles", "megatron"])
@pytest.mark.parametrize(
    "bad", ["missing", "shape", "duplicate", "mass", "nan", "empty"]
)
def test_invalid_capture_rejected_before_training(backend, bad):
    d = datum()
    if bad == "missing":
        del d.loss_fn_inputs["sampling_topk_logprobs"]
    else:
        name = (
            "sampling_topk_token_ids"
            if bad in {"shape", "duplicate", "empty"}
            else "sampling_topk_logprobs"
        )
        original = d.loss_fn_inputs[name]
        values = list(original.data)
        shape = original.shape
        if bad == "shape":
            shape = [6]
        elif bad == "duplicate":
            values[3] = 0
        elif bad == "mass":
            values[2:4] = [0.0, 0.0]
        elif bad == "nan":
            values[2] = float("nan")
        else:
            values[2:4] = [-1, -1]
        d.loss_fn_inputs[name] = TensorData(
            data=values, dtype=original.dtype, shape=shape
        )
    request = ForwardBatch(items=(ForwardItem("a", (d,)),), loss_fn="score_centering")
    with pytest.raises(ValueError):
        if backend == "miles":
            prepare_batch(request, {"a": 0})
        else:
            build_sequence_batches(request, None, max_slots=None, max_seq_length=8)


@pytest.mark.parametrize(
    "config",
    [
        {"importance_sampling_clip": 0},
        {"tail_mass_epsilon": 1},
        {"tail_mass_epsilon": float("nan")},
        {"beta": 1},
    ],
)
def test_config_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        score_centering_config(config)


def test_miles_loss_registration_preserves_other_losses_and_batch_fields():
    from spindle.backends.miles_runtime.score_centering import (
        install_score_centering_hooks,
    )

    ordinary = object()
    losses = SimpleNamespace(TINKER_LOSS_FUNCTIONS={"ppo": ordinary})
    model = SimpleNamespace(get_batch=lambda iterator, keys: keys)
    install_score_centering_hooks(
        megatron_model=model,
        tinker_losses=losses,
        logit_processors=None,
        get_parallel_state=None,
    )
    installed = model.get_batch
    install_score_centering_hooks(
        megatron_model=model,
        tinker_losses=losses,
        logit_processors=None,
        get_parallel_state=None,
    )
    assert model.get_batch is installed
    assert losses.TINKER_LOSS_FUNCTIONS["ppo"] is ordinary
    keys = ["tokens"]
    assert model.get_batch(SimpleNamespace(rollout_data={}), keys) == keys
    captured = model.get_batch(
        SimpleNamespace(
            rollout_data={
                "sampling_topk_token_ids": [[[2, 3]]],
                "sampling_topk_logprobs": [[[-1.0, -2.0]]],
            }
        ),
        keys,
    )
    assert keys == ["tokens"]
    assert captured == ["tokens", "sampling_topk_token_ids", "sampling_topk_logprobs"]
