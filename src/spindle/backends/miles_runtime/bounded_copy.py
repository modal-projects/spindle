"""Copy large files without letting dirty page cache grow unbounded.

Checkpoint volumes are mounted filesystems whose writeback is slower than a
local read, so a plain copy of hundreds of gigabytes per rank accumulates
dirty pages until the container exceeds its memory request. Syncing every
few gigabytes and dropping the cache behind the copy keeps the footprint
flat.
"""

from __future__ import annotations

import os
from pathlib import Path

CHUNK_BYTES = 256 << 20
SYNC_EVERY_BYTES = 2 << 30
_fdatasync = getattr(os, "fdatasync", os.fsync)


def copy_file_bounded(
    source: str | os.PathLike,
    target: str | os.PathLike,
    *,
    chunk_bytes: int = CHUNK_BYTES,
    sync_every_bytes: int = SYNC_EVERY_BYTES,
) -> None:
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb", buffering=0) as src, open(target, "wb", buffering=0) as dst:
        since_sync = 0
        while True:
            chunk = src.read(chunk_bytes)
            if not chunk:
                break
            dst.write(chunk)
            since_sync += len(chunk)
            if since_sync >= sync_every_bytes:
                _release(dst.fileno())
                _drop_cache(src.fileno())
                since_sync = 0
        _release(dst.fileno())
        _drop_cache(src.fileno())


def _release(fd: int) -> None:
    _fdatasync(fd)
    _drop_cache(fd)


def _drop_cache(fd: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
