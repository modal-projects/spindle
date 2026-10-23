"""Miles v0.1.1 only: reassemble full-response log probabilities for its Tinker losses
under context parallelism. Miles v0.1.2 does this itself."""

from __future__ import annotations

import torch
import torch.distributed as dist
from miles.backends.fsdp_utils import actor as fsdp_actor
from miles.backends.megatron_utils import actor as megatron_actor
from miles.backends.training_utils import cp_utils, data, loss, mm_data
from miles.backends.training_utils.loss_hub import (
    logit_processors,
    math_utils,
    tinker_losses,
)
from miles.backends.training_utils.parallel import get_parallel_state


def _pad_local_shard(
    tensor,
    total_length: int,
    response_length: int,
    *,
    qkv_format: str,
    max_seq_len,
):
    """Place this CP rank's zigzag logprob shard into a full-length response.

    Replicates the placement logic of miles'
    ``all_gather_with_cp`` without its differentiable ``dist.nn.all_reduce``.
    """
    _, _, logits_offset, _ = cp_utils.get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len
    )
    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(length: int):
        return torch.zeros(
            [length] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        padded = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        padded = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        padded = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        padded = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert padded.shape[0] == response_length, (
        f"Expected {response_length}, got {padded.shape}"
    )
    return padded


def install() -> None:
    """Reassemble full-response logprobs for Miles' Tinker loss path under CP>1."""

    original = logit_processors.get_log_probs_and_entropy
    if getattr(original, "__spindle_gathers_cp__", False):
        return

    def get_log_probs_and_entropy(
        logits,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        with_entropy=False,
        entropy_requires_grad=True,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    ):
        out = original(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            non_loss_data=non_loss_data,
            max_seq_lens=max_seq_lens,
            rollout_sampling_mask=rollout_sampling_mask,
        )
        parallel_state = get_parallel_state()
        if parallel_state.cp.size == 1 or args.allgather_cp:
            return out
        log_probs = []
        for index, (lp, total_length, response_length) in enumerate(
            zip(out["log_probs"], total_lengths, response_lengths, strict=True)
        ):
            max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
            padded = _pad_local_shard(
                lp,
                total_length,
                response_length,
                qkv_format=args.qkv_format,
                max_seq_len=max_seq_len,
            )
            summed = padded.detach().clone()
            dist.all_reduce(summed, group=parallel_state.cp.group)
            log_probs.append(padded + (summed - padded.detach()))
        out["log_probs"] = log_probs
        return out

    get_log_probs_and_entropy.__spindle_gathers_cp__ = True
    logit_processors.get_log_probs_and_entropy = get_log_probs_and_entropy

    # Callers on the multi-LoRA path that bound the name at import time must
    # be re-pointed. Miles' non-Tinker losses handle CP natively.
    for module in (
        loss,
        tinker_losses,
        megatron_actor,
        fsdp_actor,
    ):
        if module.get_log_probs_and_entropy is original:
            module.get_log_probs_and_entropy = get_log_probs_and_entropy

    # get_rollout_data slices rollout_log_probs/teacher_log_probs to the CP
    # shard for the native (non-tinker) losses. The tinker loss path pairs
    # them with full-response log_probs (gathered above) and full-length
    # advantages/loss_weights, so the slice must be disabled on this path.
    original_slice = cp_utils.slice_log_prob_with_cp
    if getattr(original_slice, "__spindle_unslices_cp__", False):
        return

    def slice_log_prob_with_cp(
        value, total_length, response_length, qkv_format, max_seq_len=None
    ):
        return value

    slice_log_prob_with_cp.__spindle_unslices_cp__ = True
    cp_utils.slice_log_prob_with_cp = slice_log_prob_with_cp
    for module in (data, mm_data, math_utils):
        if module.slice_log_prob_with_cp is original_slice:
            module.slice_log_prob_with_cp = slice_log_prob_with_cp
