from __future__ import annotations

import json
import os
import struct
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import xxhash
import zstandard
from safetensors import safe_open
from safetensors.numpy import save
from stitch.types import VersionKind, VersionManifest, VersionRef

from spindle.backends.fft_publication import (
    bulletin_board,
    discard_staged,
    publish_staged,
    write_staged_index,
)
from spindle.inference.fft_bulletin import FFTSnapshotBulletin

_INDEX = "model.safetensors.index.json"
_MAX_SHARD_BYTES = 512 << 20


@dataclass(frozen=True)
class CapturedDeltaShard:
    filename: str
    tensors: dict[str, np.ndarray]
    checksums: dict[str, str]


@dataclass(frozen=True)
class FFTCapturedDelta:
    ref: VersionRef
    parent: VersionRef
    metadata: dict[str, Any]
    shards: tuple[CapturedDeltaShard, ...]
    weight_map: dict[str, str]
    metrics: dict[str, float]


@dataclass(frozen=True)
class _PrefetchedChunk:
    name: str
    current: Any
    previous: Any
    snapshot: Any
    staging: Any
    ready: Any


class FFTDeltaWriter:
    """Encode exported weights against the last published snapshot.

    A writer rank keeps the previous Hugging Face weights of the tensors it
    exports in CPU memory, materializing each one from the base checkpoint and
    the published deltas the first time it is seen. Each capture moves those
    bytes to the GPU, finds changed bytes there, and copies only the sparse
    delta back for compression and persistence. ``writer`` selects the ranks
    that encode; by default only rank zero does. ``shard_prefix`` keeps the
    files of several writers apart in one staging directory.
    """

    def __init__(self, *, writer: bool | None = None, shard_prefix: str = "") -> None:
        self._parent: VersionRef | None = None
        self._snapshot: dict[str, Any] = {}
        self._source: _ParentSource | None = None
        self._encoder = None
        self._writer = writer
        self._shard_prefix = shard_prefix

    def is_aligned_with(self, ref: VersionRef) -> bool:
        return self._parent == ref

    def capture(
        self,
        *,
        weights: Iterable[tuple[str, Any]],
        model_id: str,
        publish_version: int,
        optimizer_step: int,
        base_model: str,
        hf_checkpoint: str,
        bulletin_root: str,
        definition_revision: str | None = None,
    ) -> FFTCapturedDelta:
        """Encode ``weights``, a collective stream of Hugging Face named
        tensors that every rank drives in lockstep and only writers read."""
        ref = VersionRef(model_id, publish_version)
        parent = VersionRef(model_id, publish_version - 1)
        board = FFTSnapshotBulletin(bulletin_root)
        writer = self._writer if self._writer is not None else dist.get_rank() == 0
        setup_error: Exception | None = None
        # All ranks must agree that the parent resolved before the export,
        # which may itself use collectives.
        if writer:
            try:
                if self._parent != parent:
                    self._source = _ParentSource(Path(hf_checkpoint), board, parent)
                    self._snapshot = {}
                    self._parent = parent
            except Exception as exc:  # noqa: BLE001 - broadcast setup errors
                setup_error = exc
        failed = torch.tensor(
            [setup_error is not None],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            if setup_error is not None:
                raise setup_error
            raise RuntimeError("delta writer setup failed on a writer rank")

        seen: set[str] = set()
        encoded: list[tuple[str, np.ndarray, str] | None] = []
        pending: deque[tuple[str, Future[tuple[np.ndarray | None, str | None]]]] = (
            deque()
        )
        workers = max(1, min(8, os.cpu_count() or 1))
        inflight_limit = min(4, workers)
        started = time.perf_counter()
        copied_bytes = 0
        prefetched: _PrefetchedChunk | None = None
        if self._encoder is None:
            self._encoder = DeltaEncoder(torch)
        encoder = self._encoder

        def collect_one() -> None:
            name, future = pending.popleft()
            compressed, checksum = future.result()
            encoded.append(
                (name, compressed, checksum)
                if compressed is not None and checksum is not None
                else None
            )

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for name, tensor in weights:
                if not writer:
                    continue
                seen.add(name)
                tensor = tensor.detach().contiguous()
                previous = self._previous(name)
                copied_bytes += int(previous.numel() * previous.element_size())

                next_prefetched = encoder.prefetch(name, tensor, previous)
                self._snapshot[name] = next_prefetched.snapshot
                if prefetched is not None:
                    pending.append(
                        (
                            prefetched.name,
                            encoder.encode_async(prefetched, pool),
                        )
                    )
                prefetched = next_prefetched
                if len(pending) >= inflight_limit:
                    collect_one()
            if prefetched is not None:
                pending.append(
                    (
                        prefetched.name,
                        encoder.encode_async(prefetched, pool),
                    )
                )
            while pending:
                collect_one()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        if self._source is not None:
            self._source.close()
            self._source = None
        missing = self._snapshot.keys() - seen
        if missing:
            raise ValueError(
                "HF tensor set changed while writing delta: "
                f"missing={sorted(missing)[:20]}"
            )

        shards, weight_map = _pack_shards(encoded, prefix=self._shard_prefix)

        metadata = {
            "version": ref.version,
            "base_version": parent.version,
            "delta_encoding": "overwrite",
            "compression_format": "zstd",
            "checksum_format": "xxh3-128",
            "base_model": base_model,
            "optimizer_step": optimizer_step,
            "parameterization": "full",
            "definition_revision": definition_revision,
        }
        self._parent = ref
        elapsed = time.perf_counter() - started
        compressed_bytes = sum(
            tensor.nbytes for shard in shards for tensor in shard.tensors.values()
        )
        return FFTCapturedDelta(
            ref=ref,
            parent=parent,
            metadata=metadata,
            shards=shards,
            weight_map=weight_map,
            metrics={
                "capture_seconds": elapsed,
                "captured_bytes": float(copied_bytes),
                "compressed_bytes": float(compressed_bytes),
            },
        )

    def persist(
        self,
        snapshot: FFTCapturedDelta,
        *,
        bulletin_root: str,
        bulletin_volume: str,
    ) -> None:
        if dist.get_rank() != 0:
            return
        board = bulletin_board(bulletin_root, bulletin_volume)
        try:
            self.stage(snapshot, board)
            publish_staged(board, ref=snapshot.ref, metadata=snapshot.metadata)
        except BaseException:
            discard_staged(board, ref=snapshot.ref, parent=snapshot.parent)
            raise

    def stage(self, snapshot: FFTCapturedDelta, board: FFTSnapshotBulletin) -> Path:
        """Write the delta's files and index into the bulletin's staging
        directory."""
        staging = self.stage_shards(snapshot, board)
        write_staged_index(
            board,
            ref=snapshot.ref,
            metadata=snapshot.metadata,
            weight_map=snapshot.weight_map,
        )
        return staging

    def stage_shards(
        self, snapshot: FFTCapturedDelta, board: FFTSnapshotBulletin
    ) -> Path:
        """Write the delta's shard files into the bulletin's staging directory;
        the index is written once the shards of every writer are there."""
        staging = board.staging_dir(snapshot.ref)
        staging.mkdir(parents=True, exist_ok=True)
        for shard in snapshot.shards:
            blob = save(shard.tensors, metadata=shard.checksums)
            (staging / shard.filename).write_bytes(blob)
        return staging

    def _previous(self, name: str):
        previous = self._snapshot.get(name)
        if previous is not None:
            return previous
        if self._source is None:
            raise ValueError(
                f"HF tensor set changed while writing delta: unexpected {name!r}"
            )
        return self._source.read(name)


class _ParentSource:
    """Tensors of a published version, read one name at a time from the base
    checkpoint and the deltas published since."""

    def __init__(
        self, base_checkpoint: Path, board: FFTSnapshotBulletin, parent: VersionRef
    ) -> None:
        self._files: dict[str, Path] = {}
        self._handles: dict[Path, Any] = {}
        try:
            index = json.loads((base_checkpoint / _INDEX).read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"base checkpoint has no valid weight index: {base_checkpoint / _INDEX}"
            ) from exc
        for name, filename in weight_map.items():
            if not str(name).startswith("mtp."):
                self._files[str(name)] = base_checkpoint / str(filename)
        if not self._files:
            raise ValueError(f"base checkpoint {base_checkpoint} contains no tensors")
        self._deltas: list[tuple[VersionRef, Path, dict[str, str]]] = []
        for version in range(1, parent.version + 1):
            ref = VersionRef(parent.run_id, version)
            source = board.resolve(ref)
            manifest = VersionManifest.from_hf_index(source, run_id=parent.run_id)
            if manifest.ref != ref:
                raise ValueError(
                    f"FFT snapshot identity mismatch: expected {ref.identity}, "
                    f"got {manifest.ref.identity}"
                )
            index = json.loads((source / _INDEX).read_text(encoding="utf-8"))
            weight_map = {
                str(name): str(filename)
                for name, filename in (index.get("weight_map") or {}).items()
            }
            if manifest.kind is VersionKind.FULL:
                if not weight_map:
                    raise ValueError(f"FFT snapshot {ref.identity} contains no tensors")
                self._files = {
                    name: source / filename
                    for name, filename in weight_map.items()
                    if not name.startswith("mtp.")
                }
                self._deltas = []
            else:
                _validate_delta_metadata(index.get("metadata") or {}, ref)
                self._deltas.append((ref, source, weight_map))

    def read(self, name: str):
        base = self._files.get(name)
        if base is None:
            raise ValueError(
                f"HF tensor set changed while writing delta: {name!r} is not in "
                "the base checkpoint"
            )
        tensor = _read_tensor(base, name)
        for ref, source, weight_map in self._deltas:
            filename = weight_map.get(name)
            if filename is None:
                continue
            delta = self._open(source / filename)
            checksum = (delta.metadata() or {}).get(name)
            if checksum is None:
                raise ValueError(
                    f"FFT delta {ref.identity} has no checksum for {name!r}"
                )
            _apply_sparse_overwrite(
                tensor,
                delta.get_tensor(name),
                checksum=checksum,
                identity=f"{ref.identity}:{name}",
            )
        return tensor

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def _open(self, path: Path):
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="numpy").__enter__()
            self._handles[path] = handle
        return handle


def _validate_delta_metadata(metadata: dict[str, Any], ref: VersionRef) -> None:
    if (
        int(metadata.get("version", -1)) != ref.version
        or int(metadata.get("base_version", -1)) != ref.version - 1
        or metadata.get("delta_encoding") != "overwrite"
        or metadata.get("compression_format") != "zstd"
        or metadata.get("checksum_format") != "xxh3-128"
    ):
        raise ValueError(f"unsupported FFT delta metadata: {ref.identity}")


class DeltaEncoder:
    """Compare GPU weights while overlapping transfers to and from the CPU."""

    def __init__(self, torch) -> None:
        self.torch = torch
        self.snapshot_to_gpu_stream = torch.cuda.Stream()
        self.result_to_cpu_stream = torch.cuda.Stream()

    def prefetch(
        self,
        name: str,
        current,
        snapshot,
    ) -> _PrefetchedChunk:
        # Snapshots stay pageable; a pinned staging buffer carries each tensor
        # to and from the GPU and returns to the caching host allocator.
        staging = self.torch.empty_like(snapshot, device="cpu", pin_memory=True)
        staging.copy_(snapshot)
        current = current.to(dtype=snapshot.dtype).view(self.torch.uint8).reshape(-1)
        staging_bytes = staging.view(self.torch.uint8).reshape(-1)
        with self.torch.cuda.stream(self.snapshot_to_gpu_stream):
            previous = staging_bytes.to(
                device=current.device,
                non_blocking=True,
            )
            ready = self.snapshot_to_gpu_stream.record_event()
        return _PrefetchedChunk(name, current, previous, snapshot, staging, ready)

    def encode_async(
        self,
        chunk: _PrefetchedChunk,
        pool: ThreadPoolExecutor,
    ) -> Future[tuple[np.ndarray | None, str | None]]:
        torch = self.torch
        compute_stream = torch.cuda.current_stream()
        compute_stream.wait_event(chunk.ready)
        chunk.previous.record_stream(compute_stream)

        mask = chunk.current != chunk.previous
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
        values = chunk.current.index_select(0, positions)
        compute_done = compute_stream.record_event()

        positions_cpu = torch.empty(
            int(positions.numel()),
            dtype=positions.dtype,
            device="cpu",
            pin_memory=True,
        )
        values_cpu = torch.empty(
            int(values.numel()),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        with torch.cuda.stream(self.result_to_cpu_stream):
            self.result_to_cpu_stream.wait_event(compute_done)
            positions_cpu.copy_(positions, non_blocking=True)
            values_cpu.copy_(values, non_blocking=True)
            staging_bytes = chunk.staging.view(torch.uint8).reshape(-1)
            staging_bytes.copy_(chunk.current, non_blocking=True)
            done = self.result_to_cpu_stream.record_event()

        positions.record_stream(self.result_to_cpu_stream)
        values.record_stream(self.result_to_cpu_stream)
        chunk.current.record_stream(self.result_to_cpu_stream)
        return pool.submit(
            _finish_sparse_overwrite,
            done,
            positions_cpu,
            values_cpu,
            staging_bytes,
            chunk.snapshot.view(torch.uint8).reshape(-1),
        )


def _finish_sparse_overwrite(
    done,
    positions_cpu,
    values_cpu,
    staging,
    snapshot,
) -> tuple[np.ndarray | None, str | None]:
    done.synchronize()
    positions = positions_cpu.numpy()
    if not positions.size:
        return None, None
    snapshot.copy_(staging)
    compressed = _compress_overwrite(
        positions,
        values_cpu.numpy(),
    )
    return compressed, xxhash.xxh3_128(snapshot.numpy()).hexdigest()


def _apply_sparse_overwrite(
    target,
    compressed: np.ndarray,
    *,
    checksum: str,
    identity: str,
) -> None:
    try:
        payload = zstandard.ZstdDecompressor().decompress(compressed.tobytes())
    except zstandard.ZstdError as exc:
        raise ValueError(f"invalid compressed FFT delta: {identity}") from exc
    if len(payload) < 4:
        raise ValueError(f"truncated FFT delta: {identity}")

    (count,) = struct.unpack_from("<I", payload)
    expected_size = 4 + 5 * count
    if len(payload) != expected_size:
        raise ValueError(
            f"invalid FFT delta size for {identity}: "
            f"expected {expected_size}, got {len(payload)}"
        )
    positions_end = 4 + 4 * count
    positions = np.frombuffer(payload, dtype="<u4", count=count, offset=4)
    values = np.frombuffer(payload, dtype=np.uint8, count=count, offset=positions_end)
    target_bytes = target.view(torch.uint8).reshape(-1).numpy()
    if positions.size and (
        int(positions[-1]) >= target_bytes.size
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError(f"invalid FFT delta positions: {identity}")
    target_bytes[positions.astype(np.intp, copy=False)] = values
    actual = xxhash.xxh3_128(target_bytes).hexdigest()
    if actual != checksum:
        raise ValueError(
            f"FFT delta checksum mismatch for {identity}: "
            f"expected {checksum}, got {actual}"
        )


def _compress_overwrite(
    positions: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if positions.size > np.iinfo(np.uint32).max:
        raise ValueError("overwrite delta has too many changed bytes")
    positions64 = np.asarray(positions, dtype=np.uint64)
    if positions64.size and int(positions64[-1]) > np.iinfo(np.uint32).max:
        raise ValueError("overwrite delta tensor exceeds uint32 byte addressing")
    positions32 = positions64.astype("<u4", copy=False)
    payload = bytearray(4 + 4 * positions32.size + values.size)
    struct.pack_into("<I", payload, 0, positions32.size)
    positions_end = 4 + 4 * positions32.size
    payload[4:positions_end] = positions32.tobytes()
    payload[positions_end:] = np.asarray(values, dtype=np.uint8).tobytes()
    return np.frombuffer(
        zstandard.ZstdCompressor(level=1).compress(payload),
        dtype=np.uint8,
    )


def _pack_shards(
    encoded: list[tuple[str, np.ndarray, str] | None],
    *,
    prefix: str = "",
) -> tuple[tuple[CapturedDeltaShard, ...], dict[str, str]]:
    shards = []
    weight_map = {}
    tensors: dict[str, np.ndarray] = {}
    checksums: dict[str, str] = {}
    shard_bytes = 0

    def flush() -> None:
        nonlocal tensors, checksums, shard_bytes
        if not tensors:
            return
        filename = f"{prefix}delta-{len(shards) + 1:05d}.safetensors"
        shards.append(CapturedDeltaShard(filename, tensors, checksums))
        weight_map.update({name: filename for name in tensors})
        tensors = {}
        checksums = {}
        shard_bytes = 0

    for item in encoded:
        if item is None:
            continue
        name, compressed, checksum = item
        if tensors and shard_bytes + compressed.nbytes > _MAX_SHARD_BYTES:
            flush()
        tensors[name] = compressed
        checksums[name] = checksum
        shard_bytes += compressed.nbytes
    flush()
    return tuple(shards), weight_map


def _read_tensor(path: Path, name: str) -> Any:
    with safe_open(path, framework="pt", device="cpu") as shard:
        return shard.get_tensor(name).clone()
