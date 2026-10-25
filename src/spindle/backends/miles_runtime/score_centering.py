"""Native Tinker losses for Miles' packed, multi-LoRA Megatron execution.

Miles owns sequence packing, adapter routing and gradient reduction. Its response
iterator supplies the original token positions even when CP splits a sequence.
Only support logits are gathered across TP; CP gathers detached output values
and leaves each rank's local gradient unchanged.
"""

from argparse import Namespace
from functools import partial

import torch
import torch.distributed as dist

from spindle.backends.score_centering import (
    REPLAY_LOSSES,
    policy_objective,
    replay_logprobs,
)


def _full_response(values, positions, length, cp):
    full = values.new_zeros(length).index_add(0, positions, values)
    if cp.size > 1:
        summed = full.detach().clone()
        dist.all_reduce(summed, group=cp.group)
        full = full + (summed - full.detach())
    return full


def loss_function(args, batch, logits, sum_of_sample_mean, *, processors):
    parallel = processors.get_parallel_state()
    # Support arithmetic owns temperature scaling. Avoid applying Miles' optional
    # true-on-policy rounding/scaling before that arithmetic.
    chunk_args = Namespace(**vars(args))
    chunk_args.true_on_policy_mode = False
    chunks = processors._iter_response_chunks(
        logits,
        args=chunk_args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=True,
    )
    losses, records = [], []
    for i, (chunk, _, response_indices) in enumerate(chunks):
        positions = torch.as_tensor(
            response_indices, dtype=torch.long, device=logits.device
        )

        def response(key, index=i, indices=positions):
            return torch.as_tensor(batch[key][index], device=logits.device)[indices]

        ids = torch.as_tensor(
            batch["rollout_sampling_mask_ids"][i],
            dtype=torch.long,
            device=logits.device,
        )
        if args.vocab_size is None or torch.any(ids >= args.vocab_size):
            raise ValueError("sampling support exceeds the true model vocabulary")
        local = {
            "token_indices": positions,
            "labels": response("target_tokens"),
            "sampling_logprobs": response("rollout_log_probs"),
            "sampling_mask_ids": ids,
            "sampling_mask_offsets": torch.as_tensor(
                batch["rollout_sampling_mask_offsets"][i],
                dtype=torch.long,
                device=logits.device,
            ),
            "loss_name": batch["loss_fn"],
            "loss_config": batch["loss_fn_config"],
        }
        if "rollout_sampling_mask_logprobs" in batch:
            local["sampling_mask_logprobs"] = torch.as_tensor(
                batch["rollout_sampling_mask_logprobs"][i], device=logits.device
            )
        sampled, correction, diagnostics = replay_logprobs(
            chunk,
            local,
            tp_rank=parallel.tp.rank,
            tp_size=parallel.tp.size,
            tp_group=parallel.tp.group,
        )
        weights = response("loss_weights") * response("loss_masks")
        objective = policy_objective(
            sampled,
            correction,
            local["sampling_logprobs"],
            batch["loss_fn"],
            batch["loss_fn_config"],
        )
        token_losses = -objective * response("advantages") * weights
        length = batch["response_lengths"][i]
        full_loss = _full_response(token_losses, positions, length, parallel.cp).sum()
        logprobs = _full_response(sampled, positions, length, parallel.cp)
        # Counts describe trained tokens, independent of advantage and client
        # normalization. Zero-advantage samples still contribute diagnostics.
        active = weights.ne(0).float()
        totals = {"tokens": active.sum()}
        totals.update(
            {key: (value * active).sum() for key, value in diagnostics.items()}
        )
        keys = list(totals)
        values = torch.stack([totals[key] for key in keys]).detach()
        if parallel.cp.size > 1:
            dist.all_reduce(values, group=parallel.cp.group)
        losses.append(full_loss)
        records.append(
            {
                "sample_index": batch["sample_indices"][i],
                "loss": full_loss.detach().cpu(),
                "logprobs": logprobs.detach().cpu(),
                "metrics": dict(zip(keys, values.cpu().tolist(), strict=True)),
            }
        )
    loss = torch.stack(losses).sum() if losses else logits.sum(dtype=torch.float32) * 0
    return loss, {"loss": loss.detach(), "per_datum": records}


def install_losses(tinker_losses, logit_processors):
    native_loss = partial(loss_function, processors=logit_processors)
    tinker_losses.TINKER_LOSS_FUNCTIONS.update(
        dict.fromkeys(REPLAY_LOSSES, native_loss)
    )
