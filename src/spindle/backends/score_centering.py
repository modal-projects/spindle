"""Score-centering loss and memory-bounded vocabulary-parallel log probabilities."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def score_centering_loss(
    logprobs,
    head_logprobs,
    sampling_logprobs,
    sampling_head_logprobs,
    head_mask,
    advantages,
    config,
):
    """Per-token loss, equations 8/14 of arXiv:2609.20807.

    Detach both the sampled importance weight and the entire head coefficient.
    This is an additive correction to the score, not advantage normalization.
    """
    p = head_logprobs.exp() * head_mask
    q = sampling_head_logprobs.exp() * head_mask
    epsilon = config.get("tail_mass_epsilon", 1e-6)
    rho = (1 - q.sum(-1)).clamp_min(epsilon) / (1 - p.sum(-1)).clamp_min(epsilon)
    clip = config.get("importance_sampling_clip")
    if clip is None:
        coefficient = q - rho.unsqueeze(-1) * p
        weighted_logprob = logprobs
    else:
        # q * min(p/q, clip) = min(p, clip*q), avoiding overflow in p/q.
        alpha = torch.minimum(torch.ones_like(rho), clip * rho)
        coefficient = torch.minimum(p, clip * q) - alpha.unsqueeze(-1) * p
        weight = (logprobs - sampling_logprobs).clamp_max(math.log(clip)).exp().detach()
        weighted_logprob = weight * logprobs
    correction = (coefficient.detach() * head_logprobs).sum(-1)
    return -advantages * (weighted_logprob - correction)


class _SelectedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ids, group, rank, size, vocab_size):
        width = logits.shape[-1]
        offset = rank * width
        local_ids = (ids - offset).clamp(0, width - 1)
        local = (ids >= offset) & (ids < offset + width) & (ids < vocab_size)
        vocab_mask = (
            torch.arange(offset, offset + width, device=logits.device) < vocab_size
        )
        values = logits.float().masked_fill(~vocab_mask, -torch.inf)
        maximum = values.amax(-1, keepdim=True)
        if size > 1:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        total = (values - maximum).exp().sum(-1, keepdim=True)
        if size > 1:
            dist.all_reduce(total, group=group)
        log_total = total.log()
        selected = values.gather(-1, local_ids).masked_fill(~local, 0)
        if size > 1:
            dist.all_reduce(selected, group=group)
        result = (selected - maximum - log_total).masked_fill(ids < 0, 0)
        ctx.save_for_backward(
            logits, local_ids, local, ids >= 0, maximum, log_total, vocab_mask
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        logits, local_ids, local, valid, maximum, log_total, vocab_mask = (
            ctx.saved_tensors
        )
        # Recompute only this chunk's probabilities; do not retain an fp32
        # [sequence, vocabulary] activation for each head token.
        probabilities = (
            (logits.float() - maximum - log_total).exp().masked_fill(~vocab_mask, 0)
        )
        grad_output = grad_output.masked_fill(~valid, 0)
        gradient = -probabilities * grad_output.sum(-1, keepdim=True)
        gradient.scatter_add_(-1, local_ids, grad_output.masked_fill(~local, 0))
        return gradient.to(logits.dtype), None, None, None, None, None


def selected_logprobs(
    logits, ids, *, group=None, rank=0, size=1, vocab_size=None, chunk_size=256
):
    """Gather several token logprobs per position without gathering vocabulary shards.

    TP ranks compute the same loss; backward returns each rank's local logit
    gradient without an extra TP reduction. Negative IDs denote padding.
    """
    logits = logits.reshape(-1, logits.shape[-1])
    vocab_size = vocab_size or size * logits.shape[-1]
    if torch.any(ids >= vocab_size):
        raise ValueError("score_centering token ID is outside the model vocabulary")
    if logits.shape[0] == 0:
        return logits.sum(-1, keepdim=True).expand(ids.shape)
    return torch.cat(
        [
            _SelectedLogprobs.apply(logit, labels, group, rank, size, vocab_size)
            for logit, labels in zip(
                logits.split(chunk_size), ids.split(chunk_size), strict=True
            )
        ]
    )
