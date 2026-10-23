from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout
from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen3_vl_model
from megatron.core import dist_checkpointing
from miles.backends.megatron_utils import model as megatron_model
from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor
from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.ray import train_actor
from miles.utils import distributed_utils
from miles.utils.replay_base import routing_replay_manager
from miles_plugins.models import qwen3_vl as miles_qwen3_vl

try:
    from miles.backends.training_utils.checkpoint import io as checkpoint_io
    from miles.backends.training_utils.loss.hub import logit_processors, tinker_losses
    from miles.backends.training_utils.replay.data import fill_replay_data

    tinker_cp = None
except ImportError:
    # Miles v0.1.1: older module paths, and its Tinker losses need tinker_cp to
    # gather log probabilities across context-parallel ranks (v0.1.2 does it itself).
    from miles.backends.training_utils import checkpoint_io
    from miles.backends.training_utils.loss_hub import logit_processors, tinker_losses
    from miles.backends.training_utils.replay_data import fill_replay_data

    from . import tinker_cp

from spindle.startup_timing import timeline

from . import expert_lora_compat, host_syncs
from .volumes import install_cross_node_checkpoint_publish, sync_checkpoint_volume
from .profiling import RankProfiler, TorchProfileConfig
from .qwen3_vl_cp import install_qwen3_vl_cp_position_ids
from .replay import install_replay_hooks


if tinker_cp is not None:
    tinker_cp.install()

install_replay_hooks(
    megatron_model=megatron_model,
    lora_model=lora_model,
    logit_processors=logit_processors,
    tinker_losses=tinker_losses,
    fill_replay_data=fill_replay_data,
    manager=routing_replay_manager,
)

install_qwen3_vl_cp_position_ids(
    bridge_model=qwen3_vl_model, miles_qwen3_vl=miles_qwen3_vl
)

host_syncs.install()


def _publish_checkpoints_across_nodes() -> None:
    install_cross_node_checkpoint_publish(checkpoint_io, checkpoint, snapshot_publisher)


_publish_checkpoints_across_nodes()


def _sync_checkpoint_volume(action: str) -> None:
    sync_checkpoint_volume(action)


def _use_distributed_timeout_for_gloo(minutes: int) -> None:
    # Miles' gloo group otherwise gets torch's 30-minute default. Ranks that load
    # weights first wait in it while ranks on cold hosts are still reading them.
    timeout = max(timedelta(minutes=minutes), default_pg_timeout)

    def init_gloo_group():
        if distributed_utils.GLOO_GROUP is None:
            distributed_utils.GLOO_GROUP = dist.new_group(
                backend="gloo", timeout=timeout
            )
        return distributed_utils.GLOO_GROUP

    train_actor.init_gloo_group = init_gloo_group


def _mark_actor_initialized() -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        timeline("trainer-rank0").mark("actor_model_loaded")


class SpindleMilesTrainRayActor(MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def init(self, args, role, **kwargs):
        _use_distributed_timeout_for_gloo(args.distributed_timeout_minutes)
        # Miles's LoRA builder inherits checkpoint MTP heads without honoring
        # enable_mtp_training. Qwen3.5 then injects an auxiliary backward loss
        # even for a client datum whose weights are all zero.
        if args.enable_mtp_training:
            result = super().init(args, role, **kwargs)
            _mark_actor_initialized()
            return result
        original = AutoBridge.to_megatron_provider

        def provider_without_mtp(bridge, *provider_args, **provider_kwargs):
            provider = original(bridge, *provider_args, **provider_kwargs)
            provider.mtp_num_layers = None
            provider.mtp_hybrid_override_pattern = None
            return provider

        AutoBridge.to_megatron_provider = provider_without_mtp
        try:
            result = super().init(args, role, **kwargs)
        finally:
            AutoBridge.to_megatron_provider = original
        _mark_actor_initialized()
        return result

    def torch_profile_start(self) -> None:
        if not TorchProfileConfig.from_env().profiles_rank(dist.get_rank()):
            self._spindle_profiler = None
            return
        self._spindle_profiler = RankProfiler()
        self._spindle_profiler.start()

    def torch_profile_stop(self, output_dir: str) -> dict | None:
        profiler = self._spindle_profiler
        if profiler is None:
            return None
        self._spindle_profiler = None
        return profiler.stop(output_dir, f"rank{dist.get_rank()}")

    def forward_backward(self, *args, **kwargs):
        return self._profiled("forward_backward", *args, **kwargs)

    def optim_step(self, *args, **kwargs):
        return self._profiled("optim_step", *args, **kwargs)

    def forward_only(self, *args, **kwargs):
        return self._profiled("forward_only", *args, **kwargs)

    def load_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        ckpt_path: str | None = None,
        load_optimizer: bool = True,
    ):
        if ckpt_path:
            _sync_checkpoint_volume("reload")
        result = super().load_slot(
            slot, rank, alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )
        if ckpt_path and expert_lora_compat.enabled():
            expert_lora_compat.restore_factory_weights(
                self.model,
                self.slot_optimizers[slot],
                ckpt_path,
                load_optimizer=load_optimizer,
            )
        return result

    def save_slot(self, *args, **kwargs):
        result = super().save_slot(*args, **kwargs)
        _sync_checkpoint_volume("commit")
        return result

    def export_slot(self, *args, **kwargs):
        result = super().export_slot(*args, **kwargs)
        _sync_checkpoint_volume("commit")
        return result

    def _profiled(self, operation: str, *args, **kwargs):
        with torch.profiler.record_function(f"spindle/{operation}"):
            result = getattr(super(), operation)(*args, **kwargs)
        self._log_peak_memory(operation)
        return result

    @staticmethod
    def _log_peak_memory(operation: str) -> None:
        gib = 1024**3
        print(
            f"spindle_memory op={operation} "
            f"allocated_gb={torch.cuda.memory_allocated() / gib:.2f} "
            f"max_allocated_gb={torch.cuda.max_memory_allocated() / gib:.2f} "
            f"reserved_gb={torch.cuda.memory_reserved() / gib:.2f} "
            f"max_reserved_gb={torch.cuda.max_memory_reserved() / gib:.2f}",
            flush=True,
        )

    def export_slot_peft(
        self,
        *,
        slot: int,
        path: str,
        rank: int,
        alpha: float,
        base_model: str,
        target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        with torch.profiler.record_function("spindle/export_slot_peft"):
            return super().export_slot_peft(
                slot=slot,
                path=path,
                rank=rank,
                alpha=alpha,
                base_model=base_model,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
            )

    def save_slot_weights(self, slot: int, path: str) -> None:
        self._save_slot_weights(slot, path)
        _sync_checkpoint_volume("commit")

    def _save_slot_weights(self, slot: int, path: str) -> None:
        weights = checkpoint._slot_weights_sharded_state_dict(self.model, slot)
        sharded = {checkpoint._WEIGHTS_KEY: weights}
        checkpoint._canonicalize_slot_keys(sharded, slot)
        checkpoint_io.write_checkpoint_dir(
            path, lambda temporary: dist_checkpointing.save(sharded, str(temporary))
        )
