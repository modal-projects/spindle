"""Checkpoint volume handling shared by the Miles trainer actors.

Each trainer node sees only its own uncommitted writes to a Modal Volume, so
checkpoints spread across nodes are assembled from the committed state and
the volume is committed or reloaded once per node.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path

import modal
import torch.distributed as dist


def _checkpoint_volume_path(path: Path) -> str | None:
    """Locate ``path`` inside the checkpoint volume, or ``None`` if outside."""
    root = _checkpoint_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return None


def _checkpoint_root() -> Path:
    return Path(os.environ.get("SPINDLE_CHECKPOINT_ROOT") or "/checkpoints")


def _write_checkpoint_dir_on_volume(
    volume_name: str,
    path: Path,
    relative: str,
    write_shards,
    metadata: dict | None,
) -> None:
    """Assemble a checkpoint from every node's committed shards."""
    tmp = path.parent / f"_tmp_{path.name}"
    tmp.mkdir(parents=True, exist_ok=True)
    tmp_relative = str(Path(relative).parent / tmp.name)
    _barrier()
    write_shards(tmp)
    if _rank() == 0 and metadata is not None:
        (tmp / "META.json").write_text(json.dumps(metadata, indent=2))
    written = _written_shard_names(tmp)
    sync_checkpoint_volume("commit")

    def publish() -> None:
        volume = modal.Volume.from_name(volume_name)
        path.mkdir(parents=True, exist_ok=True)
        volume.commit()
        local = {shard.name: shard for shard in tmp.iterdir()}
        remote = [
            entry.path
            for entry in volume.listdir(tmp_relative)
            if entry.path.rsplit("/", 1)[-1] not in local
        ]
        missing = written - set(local) - {p.rsplit("/", 1)[-1] for p in remote}
        if missing:
            raise RuntimeError(
                f"checkpoint {relative} is missing {len(missing)} shard(s) "
                f"written by peers: {', '.join(sorted(missing))}"
            )
        if remote:
            volume.copy_files(remote, relative, recursive=True)
        for name, shard in local.items():
            if shard.is_dir():
                shutil.copytree(shard, path / name, dirs_exist_ok=True)
            else:
                shutil.copy2(shard, path / name)
        print(
            f"spindle_checkpoint_publish path={relative} "
            f"local={len(local)} remote={len(remote)}",
            flush=True,
        )

    _on_rank_zero(publish)
    shutil.rmtree(tmp, ignore_errors=True)
    sync_checkpoint_volume("commit")


def install_cross_node_checkpoint_publish(checkpoint_io, *dependents) -> None:
    """Publish checkpoint shards that are spread across nodes.

    Miles writes shards to a temp dir and has rank 0 rename it into place. On a
    Modal Volume each container sees only its own uncommitted writes, so that
    rename publishes rank 0's node and silently drops every other node's shards.
    ``dependents`` are modules that bound ``write_checkpoint_dir`` at import.
    """

    original = checkpoint_io.write_checkpoint_dir
    if getattr(original, "__spindle_publishes_across_nodes__", False):
        return

    def write_checkpoint_dir(path, write_shards, metadata=None, **kwargs):
        name = os.environ.get("SPINDLE_CHECKPOINT_VOLUME")
        checkpoint_path = Path(path)
        relative = _checkpoint_volume_path(checkpoint_path)
        if name is not None and relative is None:
            print(
                f"spindle_checkpoint_publish path={path} volume={name} "
                f"reason=outside_checkpoint_root root={_checkpoint_root()}",
                flush=True,
            )
        if name is None or relative is None:
            return original(path, write_shards, metadata, **kwargs)
        return _write_checkpoint_dir_on_volume(
            name, checkpoint_path, relative, write_shards, metadata
        )

    write_checkpoint_dir.__spindle_publishes_across_nodes__ = True
    checkpoint_io.write_checkpoint_dir = write_checkpoint_dir
    for module in dependents:
        if getattr(module, "write_checkpoint_dir", None) is original:
            module.write_checkpoint_dir = write_checkpoint_dir


def _node_identity() -> str:
    """Identify the container a rank runs in."""
    return os.environ.get("MODAL_TASK_ID") or socket.gethostname()


def sync_checkpoint_volume(action: str) -> None:
    """Commit or reload the checkpoint volume once per node."""
    name = os.environ.get("SPINDLE_CHECKPOINT_VOLUME")
    if name is None:
        if "MODAL_TASK_ID" in os.environ:
            print(
                "spindle_checkpoint_volume action=skipped reason=unset",
                flush=True,
            )
        return
    if not dist.is_available() or not dist.is_initialized():
        _volume_action(name, action)
        return

    dist.barrier()
    hosts: list[str] = [""] * dist.get_world_size()
    dist.all_gather_object(hosts, _node_identity())
    representative = hosts.index(hosts[dist.get_rank()]) == dist.get_rank()
    _volume_action_on_representatives(name, action, representative)


def _written_shard_names(tmp: Path) -> set[str]:
    """All-gather the shard names each rank wrote."""
    mine = sorted(shard.name for shard in tmp.iterdir())
    if not _distributed():
        return set(mine)
    gathered: list[list[str]] = [[]] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    return {name for names in gathered for name in names}


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _on_rank_zero(work) -> None:
    """Run ``work`` on rank 0 and raise its failure on every rank.

    Ranks that skipped a failed step would otherwise carry on into the next
    collective and hang the job until the distributed timeout instead of
    surfacing the error.
    """
    failure: str | None = None
    if _rank() == 0:
        try:
            work()
        except Exception as exc:  # re-raised on every rank below
            failure = f"{type(exc).__name__}: {exc}"
    if _distributed():
        payload: list[str | None] = [failure]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]
    if failure is not None:
        raise RuntimeError(f"checkpoint publish failed on rank 0: {failure}")


def _volume_action_on_representatives(
    name: str, action: str, representative: bool
) -> None:
    """Run ``action`` on one rank per node and fail on every rank or none."""
    failure: str | None = None
    if representative:
        try:
            _volume_action(name, action)
        except Exception as exc:  # re-raised on every rank below
            failure = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"

    failures: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    reported = [f for f in failures if f is not None]
    if reported:
        raise RuntimeError(
            f"checkpoint volume {action} failed on {len(reported)} node(s): "
            + "; ".join(reported)
        )


def _volume_action(name: str, action: str) -> None:
    volume = modal.Volume.from_name(name)
    if action == "commit":
        volume.commit()
    else:
        volume.reload()
    print(f"spindle_checkpoint_volume action={action} volume={name}", flush=True)
