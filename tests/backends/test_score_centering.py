"""Compare the native loss gradient to exact finite-action expectations."""

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tinker import Datum, ModelInput, TensorData
from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

from spindle.backends import ForwardBatch, ForwardItem
from spindle.backends.megatron_runtime.common.forward_backward import (
    _loss,
    build_outputs,
    build_sequence_batches,
    pack_microbatches,
    shard_microbatches,
)
from spindle.backends.score_centering import replay_logprobs


def test_diagnostic_totals_survive_sdk_chunks_with_unequal_token_counts():
    request = ForwardBatch((ForwardItem("m", ()),), "score_centered_mis")
    chunks = []
    for count, entropy in [(1, 1.0), (9, 2.0)]:
        chunks.extend(
            build_outputs(
                request,
                {
                    0: [
                        {
                            "output_index": 0,
                            "positions": list(range(count)),
                            "logprobs": [0.0] * count,
                        }
                    ]
                },
                {
                    0: {
                        "loss": 0.0,
                        "tokens": count,
                        "sequences": 1,
                        "entropy": count * entropy,
                    }
                },
            )
        )
    combined = combine_fwd_bwd_output_results(chunks)
    assert combined.metrics["entropy:mean"] == 1.5  # SDK averages the two sequences.
    assert combined.metrics["entropy:sum"] / combined.metrics["tokens:sum"] == 1.9


def make_batch(targets, q, advantages, *, loss="score_centered_mis", config=None):
    support = [0, 2, 3]
    datum = Datum(
        ModelInput.from_ints([1] * len(targets)),
        {
            "target_tokens": targets,
            "weights": [1.0] * len(targets),
            "advantages": advantages,
            "logprobs": [math.log(q[support.index(t)]) for t in targets],
            "sampling_mask_ids": TensorData(
                data=support * len(targets), dtype="int64", shape=[3 * len(targets)]
            ),
            "sampling_mask_offsets": TensorData(
                data=list(range(0, 3 * len(targets) + 1, 3)),
                dtype="int64",
                shape=[len(targets) + 1],
            ),
            "sampling_mask_logprobs": TensorData(
                data=[math.log(x) for x in q] * len(targets),
                dtype="float32",
                shape=[3 * len(targets)],
            ),
        },
    )
    request = ForwardBatch((ForwardItem("m", (datum,)),), loss, config or {})
    sequences = build_sequence_batches(
        request, None, max_slots=None, max_seq_length=100
    )
    return pack_microbatches(sequences, max_tokens=100, pad_to_multiple=1)[0]


def native_loss(logits, batch):
    lp, correction, _ = replay_logprobs(logits, batch)
    return _loss(lp, {**batch, "score_correction": correction})[0]


@pytest.mark.parametrize("loss", ["score_centered", "score_centered_mis"])
def test_constant_advantage_has_zero_expected_gradient_even_with_masking(loss):
    q = [0.1, 0.3, 0.6]
    batch = make_batch([0, 2, 3], q, q, loss=loss)
    # The sampler almost never chooses token 0; its importance ratio exceeds 5.
    shared = torch.tensor([2.0, 100.0, 0.0, -1.0], requires_grad=True)
    objective = native_loss(shared.expand(3, -1), batch)
    objective.backward()
    torch.testing.assert_close(shared.grad, torch.zeros(4), atol=2e-7, rtol=0)


def test_mis_centering_matches_explicit_weighted_score_gradient():
    q = [0.1, 0.3, 0.6]
    batch = make_batch([0], q, [2.0])
    logits = torch.tensor([[2.0, 100.0, 0.0, -1.0]], requires_grad=True)
    native_loss(logits, batch).backward()
    p = torch.softmax(torch.tensor([2.0, 0.0, -1.0]), 0).tolist()
    ratio = [a / b for a, b in zip(p, q)]
    weights = [r if 0.2 <= r <= 5.0 else 0.0 for r in ratio]
    # Enumerate the weighted scores in the three-action categorical policy.
    expected = []
    for j in range(3):
        score = weights[0] * (float(j == 0) - p[j])
        center = sum(q[k] * weights[k] * (float(k == j) - p[j]) for k in range(3))
        expected.append(-2.0 * (score - center))
    torch.testing.assert_close(logits.grad[0, [0, 2, 3]], torch.tensor(expected))
    assert logits.grad[0, 1] == 0
    # A masked sampled token still contributes the centering correction.
    assert logits.grad.abs().sum() > 0


def test_packing_context_partition_and_dummy_padding_preserve_support():
    q = [0.2, 0.3, 0.5]
    batch = make_batch(
        [0, 2, 3], q, [1.0, -2.0, 3.0], config={"sampling_temperature": 0.7}
    )
    logits = torch.tensor(
        [[1.0, 30.0, 2.0, 3.0], [3.0, 80.0, 4.0, 1.0], [1.0, 55.0, 0.0, 2.0]],
        requires_grad=True,
    )
    native_loss(logits, batch).backward()
    expected = logits.grad.clone()
    order = torch.tensor([2, 0, 1])
    selected = {
        **batch,
        **{
            k: batch[k][order]
            for k in [
                "token_indices",
                "labels",
                "loss_mask",
                "sampling_logprobs",
                "advantages",
            ]
        },
    }
    permuted = logits.detach()[order].requires_grad_()
    native_loss(permuted, selected).backward()
    torch.testing.assert_close(permuted.grad, expected[order])
    dummy = shard_microbatches([batch], data_parallel_rank=1, data_parallel_size=2)[0]
    dummy_logits = logits.detach().clone().requires_grad_()
    native_loss(dummy_logits, dummy).backward()
    assert torch.count_nonzero(dummy_logits.grad) == 0


def test_no_drift_when_policy_and_sampler_match():
    q = [0.2, 0.3, 0.5]
    batch = make_batch([2], q, [1.0])
    logits = torch.tensor(
        [[math.log(0.2), 20.0, math.log(0.3), math.log(0.5)]], requires_grad=True
    )
    native_loss(logits, batch).backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.2, 0.0, -0.7, 0.5]]))


def test_extreme_masked_ratio_has_finite_zero_mis_gradient():
    values = torch.tensor([-0.1], requires_grad=True)
    batch = {
        "loss_name": "mis",
        "loss_config": {},
        "loss_mask": torch.ones(1),
        "sampling_logprobs": torch.tensor([-1000.0]),
        "advantages": torch.ones(1),
    }
    loss, _ = _loss(values, batch)
    loss.backward()
    assert loss == 0 and values.grad == 0


def test_rejects_missing_distribution_instead_of_approximating_it():
    datum = Datum(
        ModelInput.from_ints([1]),
        {
            "target_tokens": [2],
            "advantages": [1.0],
            "logprobs": [-1.0],
            "sampling_mask_ids": TensorData(data=[2], dtype="int64", shape=[1]),
            "sampling_mask_offsets": TensorData(data=[0, 1], dtype="int64", shape=[2]),
        },
    )
    request = ForwardBatch((ForwardItem("m", (datum,)),), "score_centered_mis")
    with pytest.raises(ValueError, match="sampling_mask_logprobs"):
        build_sequence_batches(request, None, max_slots=None, max_seq_length=10)


def _check_vocab_partition(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        batch = make_batch([0, 2], [0.1, 0.3, 0.6], [2.0, -1.0])
        full = torch.tensor(
            [[2.0, 100.0, 0.0, -1.0], [1.0, 20.0, 2.0, 0.0]], requires_grad=True
        )
        native_loss(full, batch).backward()
        shard = full.detach()[:, rank * 2 : rank * 2 + 2].clone().requires_grad_()
        lp, correction, _ = replay_logprobs(shard, batch, tp_rank=rank, tp_size=2)
        loss, _ = _loss(lp, {**batch, "score_correction": correction})
        loss.backward()
        torch.testing.assert_close(shard.grad, full.grad[:, rank * 2 : rank * 2 + 2])
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_gradient_matches_unsharded_reference(tmp_path):
    mp.spawn(_check_vocab_partition, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)
