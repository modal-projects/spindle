"""Exact score centering on the sampler's recorded (ragged) support.

The correction is E_q[w grad(log p)], including when the sampled token's
importance weight is zero. See arXiv:2609.20807, equation 11. Both w and q*w
are detached. Normalization over the recorded support also replays top-p.
"""

import math
from itertools import pairwise

import torch
import torch.distributed as dist


def validate_support(inputs, labels, weights, *, centered):
    required = {"sampling_mask_ids", "sampling_mask_offsets"}
    if centered:
        required.add("sampling_mask_logprobs")
    if not required <= inputs.keys():
        raise ValueError(f"loss requires replay tensors: {sorted(required)}")
    ids = list(inputs["sampling_mask_ids"].data)
    offsets = list(inputs["sampling_mask_offsets"].data)
    if (
        len(offsets) != len(labels) + 1
        or offsets[0] != 0
        or offsets[-1] != len(ids)
        or any(type(x) is not int or x < 0 for x in ids + offsets)
        or any(a > b for a, b in pairwise(offsets))
    ):
        raise ValueError("invalid sampling support IDs or offsets")
    probabilities = inputs.get("sampling_mask_logprobs")
    logq = list(probabilities.data) if probabilities is not None else []
    if probabilities is not None and (
        len(logq) != len(ids) or any(not math.isfinite(x) or x > 0 for x in logq)
    ):
        raise ValueError("sampling_mask_logprobs must align with support IDs")
    sampled = inputs["logprobs"].data
    for i, (start, end) in enumerate(pairwise(offsets)):
        support = ids[start:end]
        if len(set(support)) != len(support):
            raise ValueError("sampling support contains duplicate tokens")
        if weights[i] and labels[i] not in support:
            raise ValueError("every trained target must have a recorded support")
        if probabilities is not None and support:
            if not math.isclose(
                sum(math.exp(x) for x in logq[start:end]), 1, abs_tol=1e-4
            ):
                raise ValueError("sampling support probabilities must sum to one")
            if weights[i] and not math.isclose(
                logq[start + support.index(labels[i])], sampled[i], abs_tol=1e-4
            ):
                raise ValueError("sampled logprob disagrees with support probability")
    return {
        "sampling_mask_ids": torch.tensor(ids, dtype=torch.long),
        "sampling_mask_offsets": torch.tensor(offsets, dtype=torch.long),
        "sampling_mask_logprobs": torch.tensor(logq, dtype=torch.float32),
    }


def mis_weights(log_ratio, low, high):
    # Mask before exponentiation: an excluded extreme ratio must not yield inf*0.
    keep = (log_ratio >= math.log(low)) & (log_ratio <= math.log(high))
    return log_ratio.masked_fill(~keep, 0).exp().masked_fill(~keep, 0).detach()


class _SumVocabShards(torch.autograd.Function):
    """Collect disjoint logits; each TP rank evaluates the same scalar loss."""

    @staticmethod
    def forward(ctx, values, group):
        result = values.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def replay_logprobs(logits, batch, *, tp_rank=0, tp_size=1, tp_group=None):
    """Return target logprobs and E_q[w log p] with detached q*w.

    Only support logits are materialized in FP32. Positions refer to the packed
    batch before CP partitioning, so arbitrary CP order preserves alignment.
    Empty, untrained prompt/padding rows return zero.
    """
    logits = logits.reshape(-1, logits.shape[-1])
    positions = batch["token_indices"].to(logits.device)
    offsets = batch["sampling_mask_offsets"].to(logits.device)
    starts = offsets[positions]
    lengths = offsets[positions + 1] - starts
    rows = torch.repeat_interleave(
        torch.arange(len(positions), device=logits.device), lengths
    )
    local_offsets = lengths.cumsum(0) - lengths
    flat = torch.arange(rows.numel(), device=logits.device)
    source = starts[rows] + flat - local_offsets[rows]
    ids = batch["sampling_mask_ids"].to(logits.device)[source]
    vocab_start = tp_rank * logits.shape[-1]
    local_ids = ids - vocab_start
    owned = (local_ids >= 0) & (local_ids < logits.shape[-1])
    if torch.any(ids >= tp_size * logits.shape[-1]):
        raise ValueError("sampling support token exceeds model vocabulary")
    values = logits[rows, local_ids.clamp(0, logits.shape[-1] - 1)].float()
    values = values.masked_fill(~owned, 0)
    if tp_size > 1:
        values = _SumVocabShards.apply(values, tp_group)
    values = values / batch["loss_config"].get("sampling_temperature", 1.0)
    maxima = values.new_full((len(positions),), -torch.inf)
    maxima.scatter_reduce_(0, rows, values.detach(), reduce="amax")
    totals = values.new_zeros(len(positions)).index_add(
        0, rows, (values - maxima[rows]).exp()
    )
    logp = values - maxima[rows] - totals[rows].log()
    chosen = ids == batch["labels"].reshape(-1)[rows]
    sampled = values.new_zeros(len(positions)).index_add(0, rows[chosen], logp[chosen])
    correction = values.new_zeros(len(positions))
    diagnostics = {
        "entropy": correction.index_add(
            0, rows, -(logp.detach().exp() * logp.detach())
        ),
    }
    if batch["loss_name"] in {"score_centered", "score_centered_mis"}:
        logq = batch["sampling_mask_logprobs"].to(logits.device)[source]
        coefficients = logq.exp()
        diagnostics["sampler_trainer_kl"] = correction.index_add(
            0, rows, coefficients * (logq - logp.detach())
        )
        if batch["loss_name"] == "score_centered_mis":
            coefficients = coefficients * mis_weights(
                logp.detach() - logq,
                batch["loss_config"].get("mis_low", 0.2),
                batch["loss_config"].get("mis_high", 5.0),
            )
        correction = correction.index_add(0, rows, coefficients.detach() * logp)
    log_ratio = sampled.detach() - batch["sampling_logprobs"].reshape(-1)
    diagnostics["mis_masked_fraction"] = (
        mis_weights(
            log_ratio,
            batch["loss_config"].get("mis_low", 0.2),
            batch["loss_config"].get("mis_high", 5.0),
        )
        == 0
    ).float()
    return sampled, correction, diagnostics
