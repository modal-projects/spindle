"""Persistent kernel cache shared across trainer and sampler containers.

Trainers and samplers run in fresh containers, so without this every instance
compiles its kernels from scratch. Mounting one v2 Volume at ``KERNEL_CACHE_ROOT``
and pointing the compilers at it lets later containers reuse earlier compiles.
"""

from __future__ import annotations

import modal

KERNEL_CACHE_VOLUME_NAME = "spindle-kernel-cache"
KERNEL_CACHE_ROOT = "/root/.cache/kernel-cache"

kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)

KERNEL_CACHE_ENV = {
    "TRITON_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/inductor",
    "TILELANG_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/tilelang",
}

# SGLang redirects every cache it owns (Triton, Inductor, TileLang, FlashInfer,
# DeepGEMM) under this directory; kept apart from the trainer's.
SAMPLER_KERNEL_CACHE_ENV = {"SGLANG_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/sglang"}
