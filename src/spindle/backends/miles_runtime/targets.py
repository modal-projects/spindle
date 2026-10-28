"""Score alternative labels using the same model forward pass.

Miles keeps its CP sequence slicing and TP vocabulary layout. When every datum has the same
number of label columns, all columns go through Miles in one pass and share one softmax over
the vocabulary. Otherwise each column is scored separately with single-target semantics.
"""

from functools import wraps

import torch
import torch.distributed as dist


class CandidateLogProbs(torch.autograd.Function):
    """Log probabilities of K label columns from vocabulary-parallel logits [T, V/TP].

    Consumes ``logits`` in place, like Megatron's vocab-parallel cross entropy.
    """

    @staticmethod
    def forward(ctx, logits, targets, group):
        world = (
            dist.get_world_size(group)
            if group is not None and dist.is_initialized()
            else 1
        )
        start = (dist.get_rank(group) if world > 1 else 0) * logits.size(-1)
        logits_max = logits.max(dim=-1, keepdim=True).values
        if world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits.sub_(logits_max)
        outside = (targets < start) | (targets >= start + logits.size(-1))
        local = (targets - start).masked_fill(outside, 0)
        picked = logits.gather(-1, local).masked_fill(outside, 0.0)
        softmax = logits.exp_()
        sum_exp = softmax.sum(dim=-1, keepdim=True)
        if world > 1:
            dist.all_reduce(picked, group=group)
            dist.all_reduce(sum_exp, group=group)
        softmax.div_(sum_exp)
        ctx.save_for_backward(softmax, local, outside)
        return picked - sum_exp.log()

    @staticmethod
    def backward(ctx, grad):
        softmax, local, outside = ctx.saved_tensors
        grad_logits = softmax.mul_(-grad.sum(dim=-1, keepdim=True))
        grad_logits.scatter_add_(-1, local, grad.masked_fill(outside, 0.0))
        return grad_logits, None, None


def install_multi_target_hooks(tinker_losses, math_utils=None) -> None:
    original_target = tinker_losses._target_logprobs
    if math_utils is not None:
        _install_candidate_log_probs(math_utils)

    @wraps(original_target)
    def target_logprobs(args, batch, logits):
        targets = [torch.as_tensor(values) for values in batch["target_tokens"]]
        if all(target.ndim == 1 for target in targets):
            return original_target(args, batch, logits)
        widths = [target.shape[1] if target.ndim == 2 else 1 for target in targets]
        shared = {target.shape[1] for target in targets if target.ndim == 2}
        width = max(shared)
        if (
            math_utils is not None
            and shared == {width}
            and width > 1
            # Under CP a short 1-D row (DP padding) can leave a rank with no rows to score.
            and (
                args.context_parallel_size == 1
                or all(target.ndim == 2 for target in targets)
            )
            and not args.allgather_cp
            and not args.true_on_policy_mode
            and "rollout_sampling_mask_ids" not in batch
        ):
            joint_batch = dict(batch)
            joint_batch["unconcat_tokens"] = [
                torch.as_tensor(tokens)[:, None].expand(-1, width)
                for tokens in batch["unconcat_tokens"]
            ]
            joint_batch["target_tokens"] = [
                target if target.ndim == 2 else target[:, None].expand(-1, width)
                for target in targets
            ]
            scores = original_target(args, joint_batch, logits)
            return [
                score.reshape(-1, width)
                if target.ndim == 2
                else score.reshape(-1, width)[:, 0]
                for score, target in zip(scores, targets, strict=True)
            ]
        columns = []
        for column in range(max(widths)):
            column_batch = dict(batch)
            column_batch["target_tokens"] = [
                target[:, min(column, width - 1)] if target.ndim == 2 else target
                for target, width in zip(targets, widths, strict=True)
            ]
            columns.append(original_target(args, column_batch, logits))
        return [
            torch.stack([columns[column][index] for column in range(width)], dim=-1)
            if target.ndim == 2
            else columns[0][index]
            for index, (target, width) in enumerate(zip(targets, widths, strict=True))
        ]

    tinker_losses._target_logprobs = target_logprobs
    if hasattr(tinker_losses, "_local_response_masks"):
        _install_v012_hooks(tinker_losses)
    else:
        _install_v011_hooks(tinker_losses)


def _install_candidate_log_probs(math_utils) -> None:
    original = math_utils.compute_log_probs

    @wraps(original)
    def compute_log_probs(logits, tokens, process_group, **kwargs):
        if tokens.ndim == 2:
            if kwargs.get("sampling_mask") is not None or kwargs.get(
                "true_on_policy_mode"
            ):
                raise ValueError(
                    "candidate label scoring supports neither sampling masks nor true-on-policy mode"
                )
            return CandidateLogProbs.apply(logits, tokens, process_group)
        return original(logits, tokens, process_group, **kwargs)

    math_utils.compute_log_probs = compute_log_probs


def _expand_masks(masks, log_probs):
    return [
        mask.unsqueeze(-1) if log_prob.ndim == 2 else mask
        for mask, log_prob in zip(masks, log_probs, strict=True)
    ]


def _install_v011_hooks(tinker_losses) -> None:
    original_masks = tinker_losses._response_masks

    @wraps(original_masks)
    def response_masks(batch, log_probs):
        return _expand_masks(original_masks(batch, log_probs), log_probs)

    tinker_losses._response_masks = response_masks


def _install_v012_hooks(tinker_losses) -> None:
    original_masks = tinker_losses._local_response_masks

    @wraps(original_masks)
    def local_response_masks(args, batch, log_probs):
        return _expand_masks(original_masks(args, batch, log_probs), log_probs)

    original_outputs = tinker_losses._gather_per_datum_outputs

    @wraps(original_outputs)
    def gather_per_datum_outputs(args, batch, log_probs, per_datum_losses):
        # Miles reassembles 1-D log probabilities across CP ranks; do it per label column.
        widths = [
            log_prob.shape[1] if log_prob.ndim == 2 else 0 for log_prob in log_probs
        ]
        if not any(widths):
            return original_outputs(args, batch, log_probs, per_datum_losses)
        columns = [
            original_outputs(
                args,
                batch,
                [
                    log_prob[:, min(column, width - 1)] if width else log_prob
                    for log_prob, width in zip(log_probs, widths, strict=True)
                ],
                per_datum_losses,
            )
            for column in range(max(widths))
        ]
        return [
            {
                **output,
                "logprobs": torch.stack(
                    [columns[column][index]["logprobs"] for column in range(width)],
                    dim=-1,
                )
                if width
                else output["logprobs"],
            }
            for index, (output, width) in enumerate(
                zip(columns[0], widths, strict=True)
            )
        ]

    tinker_losses._local_response_masks = local_response_masks
    tinker_losses._gather_per_datum_outputs = gather_per_datum_outputs
