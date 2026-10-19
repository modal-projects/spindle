"""Two-rank CPU checks against single-process loss and gradient oracles."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from spindle.backends.score_centering import selected_logprobs  # noqa: E402
from spindle.backends.miles_runtime.score_centering import compute_score_centering_loss  # noqa: E402


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # TP: heads and sampled tokens straddle shards, include duplicates and
        # missing head entries; the last vocabulary entry is padding.
        torch.manual_seed(42)
        full = torch.randn(5, 8, requires_grad=True)
        ids = torch.tensor(
            [[0, 4, 5, -1], [6, 1, 6, 2], [3, 4, 5, 0], [-1, -1, -1, -1], [2, 2, 2, 2]]
        )
        weights = torch.randn(5, 4)
        local = full.detach()[:, rank * 4 : (rank + 1) * 4].clone().requires_grad_()
        actual = selected_logprobs(
            local,
            ids,
            group=dist.group.WORLD,
            rank=rank,
            size=2,
            vocab_size=7,
            chunk_size=2,
        )
        reference = (
            full[:, :7]
            .log_softmax(-1)
            .gather(1, ids.clamp_min(0))
            .masked_fill(ids < 0, 0)
        )
        (actual * weights).sum().backward()
        (reference * weights).sum().backward()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(local.grad, full.grad[:, rank * 4 : (rank + 1) * 4])

        # CP: each rank owns interleaved positions from two different datums.
        # Full loss values agree; each rank contributes only its own gradient.
        torch.manual_seed(10)
        full = torch.randn(1, 8, 6, requires_grad=True)
        indices = torch.tensor([rank, rank + 2, rank + 4, rank + 6])
        local = full.detach().index_select(1, indices).requires_grad_()
        batch = {
            "unconcat_tokens": [
                torch.tensor([0, 1, 2, 3, 4]),
                torch.tensor([0, 2, 3, 4, 5]),
            ],
            "total_lengths": [5, 5],
            "response_lengths": [4, 4],
            "target_tokens": [[1, 2, 3, 4], [2, 3, 4, 5]],
            "sampling_topk_token_ids": [[[0, 4]] * 4, [[2, 5]] * 4],
            "sampling_topk_logprobs": [[[-1.0, -1.2]] * 4, [[-0.8, -1.8]] * 4],
            "rollout_log_probs": [[-1.5] * 4, [-1.6] * 4],
            "advantages": [[0.0, 1.0, -1.0, 2.0], [1.0, -2.0, 1.0, 0.0]],
            "loss_masks": [[1] * 4, [1] * 4],
            "loss_fn_config": {"importance_sampling_clip": 2.0},
        }

        def chunks(logits, **kwargs):
            local_size = logits.shape[1] // 2
            for i in range(2):
                positions = [rank, rank + 2] if local_size == 2 else range(4)
                yield logits[0, i * local_size : (i + 1) * local_size], None, positions

        processor = SimpleNamespace(_iter_response_chunks=chunks)
        args = SimpleNamespace(vocab_size=6)
        tp = SimpleNamespace(group=None, rank=0, size=1)
        actual_lp, actual_losses = compute_score_centering_loss(
            args,
            batch,
            local,
            logit_processors=processor,
            parallel=SimpleNamespace(
                tp=tp, cp=SimpleNamespace(size=2, group=dist.group.WORLD)
            ),
        )
        expected_lp, expected_losses = compute_score_centering_loss(
            args,
            batch,
            full,
            logit_processors=processor,
            parallel=SimpleNamespace(tp=tp, cp=SimpleNamespace(size=1)),
        )
        for actual, expected in zip(
            actual_lp + actual_losses, expected_lp + expected_losses, strict=True
        ):
            torch.testing.assert_close(actual, expected)
        sum(actual_losses).backward()
        sum(expected_losses).backward()
        torch.testing.assert_close(local.grad, full.grad.index_select(1, indices))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires CPU distributed",
)
def test_tensor_and_context_parallel_gradients(tmp_path):
    mp.spawn(
        distributed_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2, join=True
    )
