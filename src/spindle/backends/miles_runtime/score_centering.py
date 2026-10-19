"""Keep top-k capture aligned with Miles' datum and context-parallel partitions."""

from __future__ import annotations

from argparse import Namespace
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist

from spindle.backends.score_centering import score_centering_loss, selected_logprobs
from spindle.score_centering import SCORE_CENTERING_FIELDS


def compute_score_centering_loss(args, batch, logits, *, logit_processors, parallel):
    # This loss defines p as the untruncated trainer softmax at temperature one.
    args = Namespace(**vars(args))
    args.rollout_temperature = 1.0
    args.true_on_policy_mode = False
    chunks = logit_processors._iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=True,
    )
    per_datum_losses, log_probs = [], []
    for index, (chunk, _, indices) in enumerate(chunks):
        head_ids = torch.as_tensor(
            np.asarray(batch[SCORE_CENTERING_FIELDS[0]][index]),
            device=logits.device,
            dtype=torch.long,
        )
        head_q = torch.as_tensor(
            np.asarray(batch[SCORE_CENTERING_FIELDS[1]][index]),
            device=logits.device,
            dtype=torch.float32,
        )
        labels = torch.as_tensor(
            batch["target_tokens"][index], device=logits.device, dtype=torch.long
        )
        selected_ids = torch.cat([labels.unsqueeze(-1), head_ids], dim=-1)
        indices = torch.as_tensor(list(indices), device=logits.device, dtype=torch.long)
        local = selected_logprobs(
            chunk,
            selected_ids.index_select(0, indices),
            group=parallel.tp.group,
            rank=parallel.tp.rank,
            size=parallel.tp.size,
            vocab_size=args.vocab_size,
        )
        # Gather values, preserving only the local gradient, as in the other
        # Spindle Tinker losses. This avoids a second CP sum during backward.
        full = local.new_zeros(selected_ids.shape).index_copy(0, indices, local)
        if parallel.cp.size > 1:
            gathered = full.detach().clone()
            dist.all_reduce(gathered, group=parallel.cp.group)
            full = full + (gathered - full.detach())
        target_logprobs = full[:, 0]
        per_token = score_centering_loss(
            target_logprobs,
            full[:, 1:],
            torch.as_tensor(batch["rollout_log_probs"][index], device=logits.device),
            head_q,
            head_ids >= 0,
            torch.as_tensor(batch["advantages"][index], device=logits.device),
            batch["loss_fn_config"],
        )
        mask = torch.as_tensor(batch["loss_masks"][index], device=logits.device)
        per_datum_losses.append((per_token * mask).sum())
        log_probs.append(target_logprobs)
    return log_probs, per_datum_losses


def install_score_centering_hooks(
    *, megatron_model, tinker_losses, logit_processors, get_parallel_state
):
    if "score_centering" in tinker_losses.TINKER_LOSS_FUNCTIONS:
        return
    original = megatron_model.get_batch

    @wraps(original)
    def get_batch(iterator, keys, *args, **kwargs):
        keys = [
            *keys,
            *(
                key
                for key in SCORE_CENTERING_FIELDS
                if key in iterator.rollout_data and key not in keys
            ),
        ]
        return original(iterator, keys, *args, **kwargs)

    def loss_function(args, batch, logits, sum_of_sample_mean):
        log_probs, losses = compute_score_centering_loss(
            args,
            batch,
            logits,
            logit_processors=logit_processors,
            parallel=get_parallel_state(),
        )
        return tinker_losses._sum_loss_and_outputs(batch, logits, log_probs, losses)

    megatron_model.get_batch = get_batch
    tinker_losses.TINKER_LOSS_FUNCTIONS["score_centering"] = loss_function
