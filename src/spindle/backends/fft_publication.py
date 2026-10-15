"""Promote staged full-parameter deltas into the sampler bulletin.

Trainers stage delta files under the bulletin's staging directory; these
helpers run on whichever process owns the publication lane and need neither
a GPU nor the training framework.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

import modal
from stitch.publish import publish_version as stitch_publish_version
from stitch.types import VersionRef

from spindle.inference.fft_bulletin import FFTSnapshotBulletin, FFTSnapshotStore
from spindle.providers.modal.scoped_pool import publication_pool


def bulletin_board(bulletin_root: str, bulletin_volume: str) -> FFTSnapshotBulletin:
    return FFTSnapshotBulletin(
        bulletin_root,
        commit=modal.Volume.from_name(bulletin_volume, version=2).commit,
    )


_INDEX = "model.safetensors.index.json"


def write_staged_index(
    board: FFTSnapshotBulletin,
    *,
    ref: VersionRef,
    metadata: dict[str, Any],
    weight_map: dict[str, str],
) -> None:
    """Write the index that names every staged shard of a publication."""
    staging = board.staging_dir(ref)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / _INDEX).write_text(
        json.dumps(
            {"metadata": metadata, "weight_map": weight_map},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def publish_staged(
    board: FFTSnapshotBulletin,
    *,
    ref: VersionRef,
    metadata: dict[str, Any],
) -> None:
    """Promote a staged delta into the bulletin and wake its sampler pool."""
    staging = board.staging_dir(ref)
    board.prepare(ref, staging, metadata=metadata)
    pool = publication_pool(os.environ["SPINDLE_DEFINITION_ID"], ref.run_id)
    stitch_publish_version(
        FFTSnapshotStore(board, ref.run_id),
        pool,
        str(staging),
        run_id=ref.run_id,
    )


def discard_staged(
    board: FFTSnapshotBulletin, *, ref: VersionRef, parent: VersionRef
) -> None:
    """Remove a publication that did not complete, keeping a pointer that moved."""
    current = board.read_latest(ref.run_id)
    if current is None or current == parent:
        shutil.rmtree(board.snapshot_dir(ref), ignore_errors=True)
    shutil.rmtree(board.staging_dir(ref), ignore_errors=True)
