"""Full-parameter trainer actor on Miles' Megatron actor.

Miles builds the model, its distributed Adam optimizer and the HF weight
export for us; this actor exposes Tinker's step-at-a-time operations on top
of them: gradients accumulate across forward_backward calls until
optim_step, checkpoints are Megatron torch_dist directories with optimizer
state, and sampler publications are the sparse deltas FFT samplers apply.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import finalize_model_grads
from megatron.core.utils import get_model_config, unwrap_model
from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.checkpoint import _load_checkpoint_hf
from miles.backends.megatron_utils.model import (
    run_forward_backward_pass,
    setup_train_iteration_config,
)
from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    get_hf_weight_iterator,
)
from miles.backends.megatron_utils import hf_export
from miles.backends.training_utils.checkpoint import io as checkpoint_io
from miles.backends.training_utils.data.rollout import (
    get_data_iterator,
    get_rollout_data,
)
from miles.backends.training_utils.metrics.log_utils import aggregate_train_losses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    WeightUpdatePlacement,
)
from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase

from spindle.backends.megatron_runtime.fft.delta import FFTDeltaWriter
from spindle.inference.fft_bulletin import FFTSnapshotBulletin

from .volumes import (
    install_cross_node_checkpoint_publish,
    sync_bulletin_volume,
    sync_checkpoint_volume,
)

install_cross_node_checkpoint_publish(checkpoint_io, snapshot_publisher, hf_export)

_MODEL_KEY = "model"
_OPTIMIZER_KEY = "optimizer"


def _sharded_state_dict_metadata() -> dict[str, Any]:
    """Optimizer state is saved per data-parallel shard without gathering, so a
    checkpoint restores onto the same parallel layout it was written from."""
    return {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),
    }


class SpindleMilesFullTrainRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        self._init_weight_updater_and_publisher(
            update_weights=False, publish_snapshots=False
        )
        (chunk,) = self.model
        self._chunk = chunk
        # Checkpoints are keyed on the bare model, as in Megatron's own save path.
        (self._bare_model,) = unwrap_model(self.model)
        self._delta_writer: FFTDeltaWriter | None = None
        self._hf_weights = get_hf_weight_iterator(
            self.args,
            self.model,
            required_placement=WeightUpdatePlacement(gather_pp=False),
            model_name=self._model_name(),
            quantization_config=None,
        )
        # Miles renames exports into the serving engine's parameter namespace;
        # deltas are applied by checkpoint name, so keep the checkpoint's.
        self._hf_weights._remap_hf_name = lambda name: name
        self._zero_grads()

    def _model_name(self) -> str:
        if self.args.model_name is not None:
            return self.args.model_name
        return type(self.hf_config).__name__.lower()

    def forward_backward(self, batch_id: int, rollout_data_ref) -> dict:
        return self._forward_pass(batch_id, rollout_data_ref, forward_only=False)

    def forward_only(self, batch_id: int, rollout_data_ref) -> dict:
        return self._forward_pass(batch_id, rollout_data_ref, forward_only=True)

    def _forward_pass(
        self, batch_id: int, rollout_data_ref, *, forward_only: bool
    ) -> dict:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(
                self.args, rollout_data_ref
            )
            stack.enter_context(store_get_result)
            # The Tinker bridge tags rows with LoRA slots; the full model has none.
            rollout_data.pop("adapter_slots", None)
            rollout_data.pop("n_adapters", None)
            data_iterator, num_microbatches = get_data_iterator(
                self.args, self.model, rollout_data
            )
            assert len(num_microbatches) == 1, (
                "a work unit is a single forward/backward pass"
            )
            for iterator in data_iterator:
                iterator.reset()
            self._chunk.train()
            setup_train_iteration_config(
                self.args, self.model, None, disable_optimizer=True
            )
            config = get_model_config(self._chunk)
            # Gradients stay local until optim_step reduces them once; a
            # reduce-scatter per pass would double count earlier passes.
            config.finalize_model_grads_func = lambda *args, **kwargs: None
            config.no_sync_func = None
            config.grad_sync_func = None
            dumper = DumperMegatronUtil(
                self.args, self.model, DumperPhase.FWD_BWD, rollout_id=batch_id
            )
            with self._chunk.no_sync():
                losses_reduced = run_forward_backward_pass(
                    self.args,
                    dumper,
                    data_iterator,
                    self.model,
                    num_microbatches[0],
                    num_rollouts=None,
                    forward_only=forward_only,
                )
            per_datum = [
                output
                for microbatch in losses_reduced
                for output in microbatch["per_datum"]
            ]
            dumper.finalize(self.model)
        if get_parallel_state().is_pp_last_stage:
            return {
                "metrics": aggregate_train_losses(losses_reduced, None),
                "per_datum": per_datum,
            }
        return {"metrics": {}, "per_datum": per_datum}

    def optim_step(self, adam_params: dict[str, float]) -> dict:
        self._heartbeat.bump()
        optimizer = self.optimizer
        for group in optimizer.param_groups:
            group["lr"] = adam_params["learning_rate"] * group.get("lr_mult", 1.0)
            group["betas"] = (adam_params["beta1"], adam_params["beta2"])
            group["eps"] = adam_params["eps"]
            group["weight_decay"] = adam_params["weight_decay"] * group.get(
                "wd_mult", 1.0
            )
        optimizer.config.clip_grad = adam_params["grad_clip_norm"]
        try:
            finalize_model_grads(self.model, None)
            successful, grad_norm, _ = optimizer.step()
        finally:
            self._zero_grads()
        if not successful:
            return {"skipped_nonfinite": 1.0}
        return {"grad_norm": float(grad_norm or 0.0)}

    def save_checkpoint(
        self, path: str, include_optimizer: bool, metadata: dict | None = None
    ) -> None:
        self._heartbeat.bump()

        def write(checkpoint_dir: Path) -> None:
            dist_checkpointing.save(
                self._sharded_state_dict(include_optimizer, is_loading=False),
                str(checkpoint_dir),
            )

        checkpoint_io.write_checkpoint_dir(path, write, metadata=metadata)
        sync_checkpoint_volume("commit")

    def load_checkpoint(self, path: str, restore_optimizer: bool) -> None:
        self._heartbeat.bump()
        sync_checkpoint_volume("reload")
        loaded = dist_checkpointing.load(
            self._sharded_state_dict(restore_optimizer, is_loading=True), path
        )
        self._bare_model.load_state_dict(loaded[_MODEL_KEY], strict=True)
        if restore_optimizer:
            self.optimizer.load_state_dict(loaded[_OPTIMIZER_KEY])
        else:
            self.optimizer.reload_model_params()
            reset_optimizer_states(self.optimizer)
        self._zero_grads()
        self._delta_writer = None

    def _sharded_state_dict(
        self, with_optimizer: bool, *, is_loading: bool
    ) -> dict[str, Any]:
        metadata = _sharded_state_dict_metadata()
        model_state = self._bare_model.sharded_state_dict(metadata=metadata)
        sharded = {_MODEL_KEY: model_state}
        if with_optimizer:
            sharded[_OPTIMIZER_KEY] = self.optimizer.sharded_state_dict(
                model_state, is_loading=is_loading, metadata=metadata
            )
        return sharded

    def reload_base_weights(self) -> None:
        """Return to the deployment's base checkpoint with fresh optimizer state."""
        self._heartbeat.bump()
        _load_checkpoint_hf(
            self.model, self.optimizer, self.args, self.args.hf_checkpoint
        )
        reset_optimizer_states(self.optimizer)
        self._zero_grads()
        self._delta_writer = None

    def capture_sampler_delta(
        self,
        *,
        model_id: str,
        publish_version: int,
        optimizer_step: int,
        base_model: str,
        hf_checkpoint: str,
        bulletin_root: str,
        definition_revision: str | None,
    ) -> dict[str, Any] | None:
        """Encode the weights against the last publication and stage the delta
        files in the bulletin's staging directory; rank 0 answers the manifest
        the driver publishes from."""
        self._heartbeat.bump()
        if self._delta_writer is None:
            self._delta_writer = FFTDeltaWriter(
                writer=self._writes_deltas(),
                shard_prefix=f"pp{get_parallel_state().pp.rank:02d}-",
            )
        weights = dict(self._named_actor_weights())
        exported = (
            (name, tensor)
            for bucket in self._hf_weights.iter_hf_weights(weights)
            for name, tensor in bucket
            if not name.startswith("mtp.")
        )
        snapshot = self._delta_writer.capture(
            weights=exported,
            model_id=model_id,
            publish_version=publish_version,
            optimizer_step=optimizer_step,
            base_model=base_model,
            hf_checkpoint=hf_checkpoint,
            bulletin_root=bulletin_root,
            definition_revision=definition_revision,
        )
        if self._writes_deltas():
            board = FFTSnapshotBulletin(bulletin_root)
            self._delta_writer.stage_shards(snapshot, board)
        # Each node sees only its own writes until committed; the driver
        # reloads the volume before it promotes the version.
        sync_bulletin_volume("commit")
        if not self._writes_deltas():
            return None
        return {
            "ref": snapshot.ref.identity,
            "parent": snapshot.parent.identity,
            "metadata": snapshot.metadata,
            "weight_map": snapshot.weight_map,
            "metrics": snapshot.metrics,
        }

    def _writes_deltas(self) -> bool:
        """One rank per pipeline stage encodes the stage's weights, or only
        rank zero when the iterator gathers the whole model on every stage."""
        parallel = get_parallel_state()
        if (
            parallel.tp.rank
            or parallel.intra_dp_cp.rank
            or parallel.ep.rank
            or (self._hf_weights.placement.gather_pp and parallel.pp.rank)
        ):
            return False
        return True

    def is_delta_aligned_with(self, identity: str) -> bool:
        from stitch.types import VersionRef

        return self._delta_writer is not None and self._delta_writer.is_aligned_with(
            VersionRef.parse(identity)
        )

    def forget_delta_parent(self) -> None:
        self._delta_writer = None

    def _zero_grads(self) -> None:
        self._chunk.zero_grad_buffer()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
