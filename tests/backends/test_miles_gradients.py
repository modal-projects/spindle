"""Pending gradients must survive other clients and expert/CP reductions."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spindle.backends.miles_runtime.gradients import accumulate_gradients


def chunk(*gradients):
    return SimpleNamespace(
        params_with_grad=[SimpleNamespace(main_grad=g) for g in gradients]
    )


def test_interleaved_expert_gradients_do_not_decay_on_zero_advantage_calls():
    gradients = [torch.zeros(2), torch.zeros(2)]
    model = [chunk(*gradients)]
    # Expert-DP=1, DP*CP=4: native reduction scales each new expert contribution
    # by 1/4. Retained contributions must not be scaled a second time.
    for slot, contribution in [(0, 8), (1, 12), (0, 4), (1, 0), (0, 0)]:
        with accumulate_gradients(model):
            gradients[slot].add_(contribution)
            for gradient in gradients:
                gradient.mul_(0.25)
    torch.testing.assert_close(gradients[0], torch.full((2,), 3.0))
    torch.testing.assert_close(gradients[1], torch.full((2,), 3.0))


def test_failed_pass_restores_previous_gradients():
    gradient = torch.tensor([2.0, 3.0])
    with (
        pytest.raises(RuntimeError, match="failed"),
        accumulate_gradients([chunk(gradient)]),
    ):
        gradient.add_(10)
        raise RuntimeError("failed")
    torch.testing.assert_close(gradient, torch.tensor([2.0, 3.0]))


def _parallel_accumulation(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        # Dense parameters are replicated across CP; experts are distinct on
        # each rank. Two entries represent independent adapter slots.
        dense, expert = torch.zeros(2), torch.zeros(2)
        model = [chunk(dense, expert)]
        for slot, multiplier in [(0, 1), (1, 2), (0, 3), (1, 0), (0, 0)]:
            with accumulate_gradients(model):
                dense[slot] += (rank + 1) * multiplier
                expert[slot] += (rank + 1) * multiplier
                dense.mul_(0.25)
                expert.mul_(0.25)
                dist.all_reduce(dense)
                # Expert-DP size is one, so the expert collective is identity.
        torch.testing.assert_close(dense, torch.tensor([10.0, 5.0]))
        torch.testing.assert_close(
            expert, torch.tensor([float(rank + 1), (rank + 1) / 2])
        )
    finally:
        dist.destroy_process_group()


def test_context_parallel_and_expert_gradients_accumulate_independently(tmp_path):
    mp.spawn(
        _parallel_accumulation,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=4,
        join=True,
    )
