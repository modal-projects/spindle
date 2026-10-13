"""Full-parameter training through Miles.

One Miles trainer group hosts one model. Checkpoints are Miles torch_dist
directories with optimizer state, and sampler publications are the same
sparse deltas the Megatron FFT backend produces, so FFT sampler pools apply
them without knowing which trainer wrote them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from stitch.types import VersionRef
from tinker import AdamParams, ForwardBackwardOutput, OptimStepResponse

from spindle.engine.spmd import DistributedExecutor
from spindle.errors import BackendFailed
from spindle.inference.fft_bulletin import FFTSnapshotBulletin

from .contract import Backend, ForwardBatch, ModelSpec, SamplerPublication
from .fft_publication import bulletin_board, discard_staged, publish_staged
from .miles_config import MilesBackendConfig, parse_backend_config
from .miles_lora import _adam_parameters, _commit_volume, _reload_volume
from .miles_runtime.data import build_outputs, pad_slot_rows, prepare_batch
from .miles_runtime.runtime import MilesRuntime

_SLOT = 0
CHECKPOINT_FORMAT = "miles_torch_dist"


class MilesFullBackend(Backend):
    def __init__(
        self,
        config: MilesBackendConfig,
        *,
        checkpoint_dir: Path,
        base_model: str,
        runtime: Any | None = None,
    ) -> None:
        config.validate()
        if not config.full:
            raise ValueError("MilesFullBackend requires full parameterization")
        self.config = config
        self.base_model = base_model
        self.checkpoint_dir = checkpoint_dir
        self.runtime = runtime if runtime is not None else MilesRuntime(config)
        self.model_id: str | None = None
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = False
        self._sampler_captures: dict[str, dict[str, Any] | None] = {}
        self._checkpoint_captures: dict[str, dict[str, Any]] = {}
        self._closed = False

    def accept_model(self, model_id: str, spec: ModelSpec) -> None:
        if spec.parameterization != "full" or spec.lora_config is not None:
            raise ValueError("Miles full backend requires full parameterization")
        if spec.base_model != self.base_model:
            raise ValueError(
                f"base model {spec.base_model!r} does not match deployment "
                f"{self.base_model!r}"
            )
        if self.model_id == model_id:
            return
        if self.model_id is not None:
            raise ValueError(f"FFT worker already hosts model {self.model_id}")
        if self._reset_before_accept:
            self.runtime.reload_base_weights()
            self._reset_before_accept = False
        self.model_id = model_id
        self.accumulating = False
        self.optimizer_step = 0
        self._sampler_captures.clear()
        self._checkpoint_captures.clear()

    def forward_backward(
        self, batch: ForwardBatch
    ) -> tuple[ForwardBackwardOutput, ...]:
        if not batch.items:
            return ()
        for item in batch.items:
            self._require_model(item.model_id)
        prepared = prepare_batch(
            batch,
            {self.model_id: _SLOT},
            sequence_alignment=self.config.sequence_alignment,
        )
        slot_rows = pad_slot_rows(prepared.slot_rows, self.config.data_parallel_size)
        raw_outputs = self.runtime.forward_backward(
            slot_rows,
            loss_fn=str(batch.loss_fn),
            loss_fn_config=dict(batch.loss_fn_config),
            forward_only=batch.forward_only,
        )
        if not batch.forward_only:
            self.accumulating = True
        return build_outputs(batch, prepared, raw_outputs[: len(prepared.slot_rows)])

    def optim_step(
        self, model_ids: tuple[str, ...], adam: AdamParams
    ) -> tuple[OptimStepResponse, ...]:
        if not model_ids:
            return ()
        if len(model_ids) != 1:
            raise ValueError("FFT worker steps one model at a time")
        self._require_model(model_ids[0])
        if not self.accumulating:
            raise ValueError(f"model {model_ids[0]} has no accumulated gradients")
        outcome = self.runtime.full_optim_step(_adam_parameters(adam))
        self.accumulating = False
        if "error" in outcome:
            raise BackendFailed(f"Miles optimizer failed: {outcome['error']}")
        successful = "grad_norm" in outcome
        if not successful and "skipped_nonfinite" not in outcome:
            raise BackendFailed(f"unexpected Miles optimizer outcome: {outcome}")
        self.optimizer_step += int(successful)
        metrics = {"update_successful:mean": float(successful)}
        if successful:
            metrics["grad_norm:mean"] = float(outcome["grad_norm"])
        else:
            metrics["skipped_nonfinite:sum"] = float(outcome["skipped_nonfinite"])
        return (OptimStepResponse(metrics=metrics),)

    def capture_checkpoint(
        self,
        model_id: str,
        snapshot_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> None:
        self._require_model(model_id)
        if snapshot_id in self._checkpoint_captures:
            raise ValueError(f"checkpoint snapshot already exists: {snapshot_id}")
        target = self.checkpoint_dir / destination / model_id
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        # A full checkpoint is written in place: copying it out of a capture
        # directory would double the volume traffic of its largest users.
        self.runtime.save_full_checkpoint(
            str(target / "miles"), include_optimizer=include_optimizer
        )
        metadata = {
            "schema_version": 1,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "backend": "miles",
            "miles_revision": self.runtime.revision,
            "base_model": self.base_model,
            "engine_definition_id": os.environ.get("SPINDLE_DEFINITION_ID"),
            "parameterization": {"type": "full"},
            "lora_config": None,
            "optimizer_step": self.optimizer_step,
            "has_optimizer": include_optimizer,
            "topology": self._topology(),
        }
        (target / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        self._checkpoint_captures[snapshot_id] = {
            "model_id": model_id,
            "destination": destination,
            "path": target,
        }

    def persist_checkpoint(
        self, snapshot_id: str, destination: str, *, overwrite: bool = False
    ) -> str:
        del overwrite
        try:
            capture = self._checkpoint_captures[snapshot_id]
            if capture["destination"] != destination:
                raise ValueError("snapshot destination does not match its capture")
            _commit_volume(os.environ.get("SPINDLE_CHECKPOINT_VOLUME"))
            return str(capture["path"])
        finally:
            self._checkpoint_captures.pop(snapshot_id, None)

    def load_checkpoint(
        self, model_id: str, uri: str, *, restore_optimizer: bool = False
    ) -> None:
        self._require_model(model_id)
        _reload_volume(os.environ.get("SPINDLE_CHECKPOINT_VOLUME"))
        checkpoint = Path(uri)
        try:
            metadata = json.loads(
                (checkpoint / "metadata.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Miles checkpoint: {uri}") from exc
        self._validate_checkpoint(metadata, restore_optimizer)
        self.runtime.load_full_checkpoint(
            str(checkpoint / "miles"), restore_optimizer=restore_optimizer
        )
        self.accumulating = False
        self.optimizer_step = (
            int(metadata.get("optimizer_step", 0)) if restore_optimizer else 0
        )
        self._sampler_captures.clear()

    def capture_sampler_snapshot(
        self, model_id: str, capture_id: str, requested_version: int
    ) -> SamplerPublication:
        # FFT publications are contiguous; the requested version is the
        # client's operation sequence and may skip numbers.
        del requested_version
        self._require_model(model_id)
        if self.accumulating:
            raise ValueError("cannot publish with accumulated gradients")
        if capture_id in self._sampler_captures:
            raise ValueError(f"sampler capture already exists: {capture_id}")
        bulletin_root = os.environ["SPINDLE_BULLETIN_ROOT"]
        bulletin = FFTSnapshotBulletin(bulletin_root)
        current = bulletin.read_latest(model_id)
        durable_version = current.version if current is not None else 0
        published_step = (
            int(bulletin.metadata(current).get("optimizer_step", 0))
            if current is not None
            else 0
        )
        if (
            current is not None
            and current.version > 0
            and self.optimizer_step == published_step
            and self.runtime.full_delta_aligned_with(current.identity)
        ):
            self._sampler_captures[capture_id] = None
            return SamplerPublication(
                publish_version=current.version,
                base_model=self.base_model,
                optimizer_step=self.optimizer_step,
            )
        publish_version = durable_version + 1
        manifest = self.runtime.capture_full_delta(
            model_id=model_id,
            publish_version=publish_version,
            optimizer_step=self.optimizer_step,
            base_model=self.base_model,
            hf_checkpoint=self.config.hf_checkpoint,
            bulletin_root=bulletin_root,
            definition_revision=os.environ.get("SPINDLE_DEFINITION_REVISION"),
        )
        self._sampler_captures[capture_id] = manifest
        return SamplerPublication(
            publish_version=publish_version,
            base_model=self.base_model,
            optimizer_step=self.optimizer_step,
        )

    def publish_sampler_snapshot(self, capture_id: str) -> None:
        manifest = self._sampler_captures.pop(capture_id)
        if manifest is None:
            return
        ref = VersionRef.parse(manifest["ref"])
        parent = VersionRef.parse(manifest["parent"])
        board = bulletin_board(
            os.environ["SPINDLE_BULLETIN_ROOT"], os.environ["SPINDLE_BULLETIN_VOLUME"]
        )
        try:
            publish_staged(board, ref=ref, metadata=manifest["metadata"])
        except BaseException:
            discard_staged(board, ref=ref, parent=parent)
            raise

    def unload_model(self, model_id: str) -> None:
        if self.model_id != model_id:
            return
        self.model_id = None
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = True
        self._sampler_captures.clear()
        self._checkpoint_captures.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runtime.close()

    def _require_model(self, model_id: str) -> None:
        if self.model_id != model_id:
            raise KeyError(f"unknown model: {model_id}")

    def _topology(self) -> dict[str, int]:
        return {
            "world_size": self.config.world_size,
            "data_parallel_size": self.config.data_parallel_size,
            "tensor_model_parallel_size": self.config.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.config.pipeline_model_parallel_size,
            "context_parallel_size": self.config.context_parallel_size,
            "expert_model_parallel_size": self.config.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.config.expert_tensor_parallel_size,
        }

    def _validate_checkpoint(
        self, metadata: dict[str, Any], restore_optimizer: bool
    ) -> None:
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported Miles checkpoint schema")
        if metadata.get("backend") != "miles":
            raise ValueError("checkpoint was not created by the Miles backend")
        if metadata.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported Miles checkpoint format")
        if metadata.get("miles_revision") != self.runtime.revision:
            raise ValueError("checkpoint Miles revision does not match the deployment")
        if metadata.get("base_model") != self.base_model:
            raise ValueError("checkpoint base model does not match the deployment")
        if (metadata.get("parameterization") or {}).get("type") != "full":
            raise ValueError("checkpoint was not created by full-parameter training")
        if metadata.get("topology") != self._topology():
            raise ValueError("checkpoint topology does not match deployment")
        if restore_optimizer and not metadata.get("has_optimizer"):
            raise ValueError("checkpoint does not include optimizer state")


def build_executor() -> DistributedExecutor:
    config, checkpoint_dir, _ = parse_backend_config(
        json.loads(os.environ["SPINDLE_BACKEND_CONFIG"])
    )
    backend = MilesFullBackend(
        config,
        checkpoint_dir=checkpoint_dir,
        base_model=os.environ["SPINDLE_BASE_MODEL"],
    )
    return DistributedExecutor(backend)
