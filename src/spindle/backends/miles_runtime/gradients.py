"""Accumulate independently reduced Miles work units without reducing history."""

from contextlib import contextmanager

import torch


@contextmanager
def accumulate_gradients(model):
    """Keep previous work units outside this pass's gradient collectives.

    Miles retains ``main_grad`` between Tinker calls, but Megatron's expert
    reduction is not idempotent when expert-DP and DP-with-CP sizes differ.
    Repeated reduction would rescale old expert gradients, including another
    client's pending gradients. Reduce only the new contribution, then add the
    already-reduced history back. This also avoids repeating TP reductions on
    previously accumulated gradients.
    """
    with torch.no_grad():
        saved = [
            (parameter.main_grad, parameter.main_grad.clone())
            for chunk in model
            for parameter in chunk.params_with_grad
        ]
        for gradient, _ in saved:
            gradient.zero_()
    try:
        yield
    except BaseException:
        with torch.no_grad():
            for gradient, previous in saved:
                gradient.copy_(previous)
        raise
    else:
        with torch.no_grad():
            for gradient, previous in saved:
                gradient.add_(previous)
