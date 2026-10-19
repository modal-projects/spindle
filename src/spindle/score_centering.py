"""Input validation for top-k score centering (arXiv:2609.20807)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

SCORE_CENTERING_FIELDS = ("sampling_topk_token_ids", "sampling_topk_logprobs")


def score_centering_config(config: Mapping[str, float]) -> dict[str, float]:
    allowed = {"importance_sampling_clip", "tail_mass_epsilon"}
    if set(config) - allowed:
        raise ValueError(
            f"unsupported score_centering loss_fn_config: {sorted(set(config) - allowed)}"
        )
    values = {key: float(value) for key, value in config.items()}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError(
            "score_centering loss_fn_config values must be positive and finite"
        )
    if values.get("tail_mass_epsilon", 1e-6) >= 1:
        raise ValueError("tail_mass_epsilon must be less than one")
    return values


def score_centering_inputs(inputs, active) -> dict[str, np.ndarray]:
    """Validate dense [target positions, k] tensors before distributed execution.

    -1 token IDs mark unused entries (including prompt/padding rows). Probabilities
    are normalized over the sampler vocabulary, never renormalized over the head.
    """
    for name in ("advantages", "logprobs"):
        if any(not math.isfinite(value) for value in inputs[name].data):
            raise ValueError(f"{name} must be finite for score_centering")
    arrays = {}
    for name, dtype in zip(SCORE_CENTERING_FIELDS, ("int64", "float32"), strict=True):
        tensor = inputs.get(name)
        if tensor is None:
            raise ValueError(f"{name} is required for score_centering")
        values = tensor.data
        shape = list(tensor.shape or ())
        if (
            tensor.dtype != dtype
            or tensor.sparse_crow_indices is not None
            or len(shape) != 2
            or shape[0] != len(active)
            or shape[1] < 1
            or math.prod(shape) != len(values)
        ):
            raise ValueError(
                f"{name} must be a dense {dtype} tensor of shape [input_tokens, k]"
            )
        if dtype == "int64" and any(
            type(v) is not int or v < -1 or v > 2**31 - 1 for v in values
        ):
            raise ValueError(
                "sampling_topk_token_ids must contain token IDs or -1 padding"
            )
        arrays[name] = np.asarray(values, dtype=dtype).reshape(shape)
    ids, logprobs = (arrays[key] for key in SCORE_CENTERING_FIELDS)
    if ids.shape != logprobs.shape:
        raise ValueError("sampling top-k IDs and logprobs must have the same shape")
    valid = ids >= 0
    if not np.isfinite(logprobs).all() or np.any(logprobs > 0):
        raise ValueError("sampling_topk_logprobs must be finite and non-positive")
    if np.any(np.asarray(active, dtype=bool) & ~valid.any(axis=1)):
        raise ValueError(
            "every active score_centering position needs sampler top-k probabilities"
        )
    mass = np.where(valid, np.exp(logprobs.astype(np.float64)), 0).sum(axis=1)
    if np.any(mass > 1 + 1e-5):
        raise ValueError("sampler top-k probability mass exceeds one")
    ordered = np.sort(ids, axis=1)
    if np.any((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])):
        raise ValueError("sampler top-k token IDs must be unique at each position")
    return arrays
