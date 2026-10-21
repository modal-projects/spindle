"""Score alternative labels using the same model forward pass.

Delegate each label column to Miles so its TP vocabulary reductions, CP sequence
slicing, and sampling masks retain the same semantics as single-target scoring.
"""

from functools import wraps

import torch


def install_multi_target_hooks(tinker_losses) -> None:
    original_target = tinker_losses._target_logprobs

    @wraps(original_target)
    def target_logprobs(args, batch, logits):
        targets = [torch.as_tensor(values) for values in batch["target_tokens"]]
        if all(target.ndim == 1 for target in targets):
            return original_target(args, batch, logits)
        widths = [target.shape[1] if target.ndim == 2 else 1 for target in targets]
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

    original_masks = tinker_losses._response_masks

    @wraps(original_masks)
    def response_masks(batch, log_probs):
        return [
            mask.unsqueeze(-1) if log_prob.ndim == 2 else mask
            for mask, log_prob in zip(
                original_masks(batch, log_probs), log_probs, strict=True
            )
        ]

    tinker_losses._target_logprobs = target_logprobs
    tinker_losses._response_masks = response_masks
