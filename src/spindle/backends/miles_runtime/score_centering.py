"""Miles response packing with the shared exact replay objective."""

from argparse import Namespace
from functools import partial

import torch
import torch.distributed as dist

from spindle.backends.megatron_runtime.common.score_centering import (
    mis_weights,
    replay_logprobs,
)


def native_loss(
    args,
    batch,
    logits,
    sum_of_sample_mean,
    *,
    name,
    processors,
    losses,
    parallel,
    pad_local_shard,
):
    if args.allgather_cp:
        raise ValueError("Miles score-centering losses do not support allgather_cp")
    config = batch.get("loss_fn_config") or {}
    args = Namespace(**vars(args))
    args.rollout_temperature = 1.0
    # _iter_response_chunks otherwise scales logits before replay_logprobs does.
    args.true_on_policy_mode = False
    state = parallel()
    labels = [
        torch.cat(
            [
                tokens[: len(tokens) - len(targets)],
                torch.as_tensor(targets, device=tokens.device, dtype=tokens.dtype),
            ]
        )
        for tokens, targets in zip(
            batch["unconcat_tokens"], batch["target_tokens"], strict=True
        )
    ]
    chunks = processors._iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=labels,
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=True,
    )
    log_probs, per_losses, diagnostics = [], [], []
    for i, (chunk, _, positions) in enumerate(chunks):

        def tensor(values):
            return torch.as_tensor(values, device=logits.device)

        targets = tensor(batch["target_tokens"][i]).long()
        rollout = tensor(batch["rollout_log_probs"][i])
        if "rollout_sampling_mask_ids" in batch:
            ids = tensor(batch["rollout_sampling_mask_ids"][i]).long()
            if args.vocab_size is None or torch.any(ids >= args.vocab_size):
                raise ValueError("sampling support token exceeds model vocabulary")
            replay = {
                "token_indices": tensor(list(positions)).long(),
                "sampling_mask_ids": ids,
                "sampling_mask_offsets": tensor(
                    batch["rollout_sampling_mask_offsets"][i]
                ).long(),
                "sampling_mask_logprobs": tensor(
                    batch.get("rollout_sampling_mask_logprobs", [[]] * len(labels))[i]
                ),
                "labels": targets[tensor(list(positions)).long()],
                "sampling_logprobs": rollout[tensor(list(positions)).long()],
                "loss_config": config,
                "loss_name": name,
            }
            lp, correction, diag = replay_logprobs(
                chunk,
                replay,
                tp_rank=state.tp.rank,
                tp_size=state.tp.size,
                tp_group=state.tp.group,
            )
        else:
            if name != "mis":
                raise ValueError("centered losses require recorded sampling supports")
            lp, entropy = processors.calculate_log_probs_and_entropy(
                chunk,
                targets[tensor(list(positions)).long()],
                state.tp.group,
                with_entropy=True,
                entropy_requires_grad=False,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=False,
                vocab_size=args.vocab_size,
                temperature=config.get("sampling_temperature", 1.0),
            )
            lp = lp.squeeze(-1)
            correction = torch.zeros_like(lp)
            diag = {
                "entropy": entropy,
                "mis_masked_fraction": (
                    mis_weights(
                        lp.detach() - rollout[tensor(list(positions)).long()],
                        config.get("mis_low", 0.2),
                        config.get("mis_high", 5.0),
                    )
                    == 0
                ).float(),
            }

        def gather(value, index=i):
            if state.cp.size == 1:
                return value
            padded = pad_local_shard(
                value,
                batch["total_lengths"][index],
                batch["response_lengths"][index],
                qkv_format=args.qkv_format,
                max_seq_len=batch["max_seq_lens"][index]
                if batch.get("max_seq_lens") is not None
                else None,
            )
            summed = padded.detach().clone()
            dist.all_reduce(summed, group=state.cp.group)
            return padded + (summed - padded.detach())

        lp, correction = gather(lp), gather(correction)
        mask = tensor(batch["loss_masks"][i]).to(lp)
        advantage = tensor(batch["advantages"][i]).to(lp)
        weight = (
            1
            if name == "score_centered"
            else mis_weights(
                lp.detach() - rollout,
                config.get("mis_low", 0.2),
                config.get("mis_high", 5.0),
            )
        )
        objective = advantage * (weight * lp - correction)
        log_probs.append(lp)
        per_losses.append(-(objective * mask).sum())
        diagnostics.append(
            {
                **{
                    key: (gather(value.detach()) * mask).sum().detach().cpu()
                    for key, value in diag.items()
                },
                "tokens": mask.sum().detach().cpu(),
            }
        )
    result, outputs = losses._sum_loss_and_outputs(batch, logits, log_probs, per_losses)
    for output, diag in zip(outputs["per_datum"], diagnostics, strict=True):
        output["diagnostics"] = diag
    return result, outputs


def install_score_centering(
    *, tinker_losses, logit_processors, get_parallel_state, pad_local_shard
):
    for name in ("mis", "score_centered", "score_centered_mis"):
        tinker_losses.TINKER_LOSS_FUNCTIONS[name] = partial(
            native_loss,
            name=name,
            processors=logit_processors,
            losses=tinker_losses,
            parallel=get_parallel_state,
            pad_local_shard=pad_local_shard,
        )
