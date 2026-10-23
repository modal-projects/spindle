"""Spindle command backend using a frozen native TorchTitan decoder."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import threading
import tempfile
import uuid

import modal
from safetensors.torch import save_file
from stitch.types import VersionRef
import torch
from torch.utils.checkpoint import checkpoint
from tinker import OptimStepResponse

from spindle.engine.spmd import DistributedExecutor
from spindle.inference.bulletin import SnapshotBulletin

from .contract import SamplerPublication
from .megatron_runtime.common.forward_backward import _loss, _loss_config
from .miles_runtime.data import prepare_batch, build_outputs
from .torchtitan_config import TorchTitanConfig
from .torchtitan_runtime.lora import selected_logprobs
from .torchtitan_runtime.model import build_model


@dataclass
class AdapterState:
    slot: str
    spec: object
    optimizer: torch.optim.Optimizer
    step: int = 0
    accumulating: bool = False


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


class TorchTitanBackend:
    def __init__(
        self, config, *, base_model, model, adapter_batch, projections, checkpoint_dir
    ):
        self.config = config
        self.base_model = base_model
        self.model = model
        self.adapter_batch = adapter_batch
        self.projections = projections
        self.device = next(model.parameters()).device
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.jobs = {}
        self.captures = {}
        self.sampler_captures = {}
        self.io_lock = threading.RLock()

    def parameters_for(self, slot):
        return [
            p
            for module in self.projections.values()
            for p in (module.a[slot], module.b[slot])
        ]

    def accept_model(self, model_id, spec):
        if model_id in self.jobs:
            if self.jobs[model_id].spec != spec:
                raise ValueError("model already exists with a different specification")
            return
        if spec.base_model != self.base_model or spec.parameterization != "lora":
            raise ValueError("TorchTitan requires the deployment's LoRA base model")
        lora = spec.lora_config
        if not 0 < lora.rank <= self.config.max_lora_rank:
            raise ValueError("LoRA rank exceeds the deployment's supported range")
        configured = set(self.config.target_modules)
        if (lora.train_attn, lora.train_mlp, lora.train_unembed) != (
            bool(configured & {"q_proj", "k_proj", "v_proj", "o_proj"}),
            bool(configured & {"gate_proj", "up_proj", "down_proj"}),
            False,
        ):
            raise ValueError(
                "LoRA train flags must match deployment targets; unembedding is frozen"
            )
        available = set(map(str, range(self.config.max_lora_slots))) - {
            job.slot for job in self.jobs.values()
        }
        if not available:
            raise ValueError("no free TorchTitan LoRA slots")
        slot = min(available, key=int)
        generator = torch.Generator(device=self.device)
        if lora.seed is None:
            generator.seed()
        else:
            generator.manual_seed(lora.seed)
        added = []
        try:
            for module in self.projections.values():
                module.add_adapter(slot, lora.rank, self.config.lora_alpha, generator)
                added.append(module)
            optimizer = torch.optim.AdamW(
                self.parameters_for(slot), lr=1e-4, fused=self.device.type == "cuda"
            )
            self.jobs[model_id] = AdapterState(slot, spec, optimizer)
        except BaseException:
            for module in added:
                module.remove_adapter(slot)
            raise

    def forward_backward(self, batch):
        if not batch.items:
            return ()
        loss_config = _loss_config(batch.loss_fn, dict(batch.loss_fn_config))
        prepared = prepare_batch(
            batch, {key: int(job.slot) for key, job in self.jobs.items()}
        )
        for _, row in prepared.slot_rows:
            if row["target_len"] > min(
                self.config.max_context_length, self.config.max_batch_tokens
            ):
                raise ValueError(
                    "sequence exceeds TorchTitan context or microbatch token budget"
                )
            if "routed_experts" in row or "sampling_mask_ids" in row:
                raise ValueError(
                    "TorchTitan does not support router replay or sampling masks"
                )
        outputs = [None] * len(prepared.slot_rows)
        # Similar lengths limit padding. Keep locations for stable SDK ordering.
        pending = sorted(
            enumerate(prepared.slot_rows), key=lambda item: item[1][1]["target_len"]
        )
        while pending:
            count = 1
            while count < min(len(pending), self.config.max_batch_rows):
                longest = pending[count][1][1]["target_len"]
                if longest * (count + 1) > self.config.max_batch_tokens:
                    break
                count += 1
            microbatch, pending = pending[:count], pending[count:]
            width = max(row["target_len"] for _, (_, row) in microbatch)
            tokens = torch.zeros((count, width), dtype=torch.long, device=self.device)
            targets = torch.zeros_like(tokens)
            for i, (_, (_, row)) in enumerate(microbatch):
                length = row["target_len"]
                tokens[i, :length] = torch.tensor(
                    row["tokens"][:-1], device=self.device
                )
                targets[i, :length] = torch.tensor(
                    row["target_tokens"], device=self.device
                )
            positions = torch.arange(width, device=self.device).repeat(count)
            self.adapter_batch.slots = tuple(str(slot) for _, (slot, _) in microbatch)
            try:
                with torch.no_grad() if batch.forward_only else nullcontext():
                    metadata = self.model._get_attention_metadata(positions)
                    hidden = self.model(
                        tokens.flatten(),
                        positions=positions,
                        attention_metadata=metadata,
                    )
                    logprobs = []
                    for start in range(0, hidden.shape[0], self.config.loss_chunk_size):
                        args = (
                            hidden[start : start + self.config.loss_chunk_size],
                            self.model.lm_head.weight,
                            targets.flatten()[
                                start : start + self.config.loss_chunk_size
                            ],
                        )
                        logprobs.append(
                            selected_logprobs(*args)
                            if batch.forward_only
                            else checkpoint(
                                selected_logprobs,
                                *args,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                        )
                    logprobs = torch.cat(logprobs).view(count, width)
                    losses = []
                    for i, (index, (_, row)) in enumerate(microbatch):
                        lp = logprobs[i, : row["target_len"]]
                        fields = {
                            "loss_name": batch.loss_fn,
                            "loss_config": loss_config,
                            "loss_mask": torch.tensor(
                                row.get("weights", [1.0] * row["target_len"]),
                                device=self.device,
                            ),
                        }
                        if batch.loss_fn != "cross_entropy":
                            fields.update(
                                {
                                    key: torch.tensor(row[key], device=self.device)
                                    for key in ("advantages", "sampling_logprobs")
                                }
                            )
                        loss, _ = _loss(lp, fields)
                        losses.append(loss)
                        outputs[index] = {
                            "logprobs": lp.detach().cpu().tolist(),
                            "loss": loss.detach().item(),
                        }
                    if not batch.forward_only:
                        torch.stack(losses).sum().backward()
            finally:
                self.adapter_batch.slots = ()
        if not batch.forward_only:
            for item in batch.items:
                self.jobs[item.model_id].accumulating = True
        return build_outputs(batch, prepared, outputs)

    def optim_step(self, model_ids, adam):
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("duplicate optimizer model IDs")
        values = [
            adam.learning_rate,
            adam.beta1,
            adam.beta2,
            adam.eps,
            adam.weight_decay,
            adam.grad_clip_norm,
        ]
        if not all(math.isfinite(value) for value in values) or not (
            adam.learning_rate >= 0
            and 0 <= adam.beta1 < 1
            and 0 <= adam.beta2 < 1
            and adam.eps > 0
            and adam.weight_decay >= 0
            and adam.grad_clip_norm >= 0
        ):
            raise ValueError("invalid Adam parameters")
        states = [self.jobs[model_id] for model_id in model_ids]
        if any(not state.accumulating for state in states):
            raise ValueError("optimizer step requires accumulated gradients")
        outputs = []
        for state in states:
            parameters = self.parameters_for(state.slot)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, adam.grad_clip_norm or float("inf")
            )
            successful = bool(torch.isfinite(norm))
            if successful:
                for group in state.optimizer.param_groups:
                    group.update(
                        lr=adam.learning_rate,
                        betas=(adam.beta1, adam.beta2),
                        eps=adam.eps,
                        weight_decay=adam.weight_decay,
                    )
                state.optimizer.step()
                state.step += 1
            state.optimizer.zero_grad(set_to_none=True)
            state.accumulating = False
            outputs.append(
                OptimStepResponse(
                    metrics={
                        "update_successful:mean": float(successful),
                        **(
                            {"grad_norm:mean": float(norm)}
                            if successful
                            else {"skipped_nonfinite:sum": 1.0}
                        ),
                    }
                )
            )
        return tuple(outputs)

    def _adapter_weights(self, state):
        rank = state.spec.lora_config.rank
        return {
            key: value.detach().cpu().contiguous().clone()
            for name, module in self.projections.items()
            for key, value in module.peft_tensors(name, state.slot, rank).items()
        }

    def capture_checkpoint(
        self, model_id, snapshot_id, *, destination, include_optimizer
    ):
        if snapshot_id in self.captures:
            raise ValueError("checkpoint capture already exists")
        state = self.jobs[model_id]
        self.captures[snapshot_id] = (
            destination,
            {
                "format": "spindle_torchtitan_lora_v1",
                "base_model": self.base_model,
                "lora_config": state.spec.lora_config.model_dump(),
                "alpha": self.config.lora_alpha,
                "step": state.step,
                "max_lora_rank": self.config.max_lora_rank,
                "weights": self._adapter_weights(state),
                "optimizer": cpu_copy(state.optimizer.state_dict())
                if include_optimizer
                else None,
                "gradients": cpu_copy([p.grad for p in self.parameters_for(state.slot)])
                if include_optimizer
                else None,
                "accumulating": state.accumulating if include_optimizer else False,
            },
        )

    def _checkpoint_path(self, path):
        result = Path(path)
        if not result.is_absolute():
            result = self.checkpoint_dir / result
        result = result.resolve()
        if (
            not result.is_relative_to(self.checkpoint_dir)
            or result == self.checkpoint_dir
        ):
            raise ValueError("checkpoint path must be inside checkpoint_dir")
        return result

    def persist_checkpoint(self, snapshot_id, destination, *, overwrite=False):
        saved_destination, capture = self.captures[snapshot_id]
        if saved_destination != destination:
            raise ValueError("checkpoint destination differs from capture")
        target = self._checkpoint_path(destination)
        with self.io_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}-{uuid.uuid4().hex}")
            try:
                if target.exists() and not overwrite:
                    raise FileExistsError(target)
                temporary.mkdir()
                torch.save(capture, temporary / "adapter.pt")
                if target.exists():
                    shutil.rmtree(target)
                os.replace(temporary, target)
                self._volume("commit")
                del self.captures[snapshot_id]
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return str(target)

    def load_checkpoint(self, model_id, uri, *, restore_optimizer=False):
        state = self.jobs[model_id]
        with self.io_lock:
            self._volume("reload")
            saved = torch.load(
                self._checkpoint_path(uri) / "adapter.pt",
                map_location="cpu",
                weights_only=True,
            )
        if (
            saved["format"] != "spindle_torchtitan_lora_v1"
            or saved["base_model"] != self.base_model
        ):
            raise ValueError("checkpoint backend or base model does not match")
        expected = state.spec.lora_config.model_dump()
        if (
            any(
                saved["lora_config"][key] != expected[key]
                for key in ("rank", "train_attn", "train_mlp", "train_unembed")
            )
            or saved["alpha"] != self.config.lora_alpha
        ):
            raise ValueError("checkpoint LoRA configuration does not match")
        weights = self._adapter_weights(state)
        if weights.keys() != saved["weights"].keys() or any(
            value.shape != saved["weights"][key].shape for key, value in weights.items()
        ):
            raise ValueError("checkpoint adapter projections do not match")
        if restore_optimizer and saved["optimizer"] is None:
            raise ValueError("checkpoint does not contain optimizer state")
        if restore_optimizer and saved["max_lora_rank"] != self.config.max_lora_rank:
            raise ValueError("optimizer checkpoint requires the same maximum LoRA rank")
        rank = expected["rank"]
        with torch.no_grad():
            for name, module in self.projections.items():
                module.a[state.slot].zero_()
                module.b[state.slot].zero_()
                for key, tensor in module.peft_tensors(name, state.slot, rank).items():
                    tensor.copy_(saved["weights"][key])
        state.optimizer.zero_grad(set_to_none=True)
        if restore_optimizer:
            state.optimizer.load_state_dict(saved["optimizer"])
            for parameter, gradient in zip(
                self.parameters_for(state.slot), saved["gradients"], strict=True
            ):
                parameter.grad = (
                    None if gradient is None else gradient.to(parameter.device)
                )
        else:
            state.optimizer.state.clear()
        state.step = saved["step"] if restore_optimizer else 0
        state.accumulating = saved["accumulating"] if restore_optimizer else False

    def capture_sampler_snapshot(self, model_id, capture_id, requested_version):
        if capture_id in self.sampler_captures:
            raise ValueError("sampler capture already exists")
        state = self.jobs[model_id]
        self.sampler_captures[capture_id] = (
            model_id,
            requested_version,
            self._adapter_weights(state),
            {
                "base_model_name_or_path": self.base_model,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": state.spec.lora_config.rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": list(self.config.target_modules),
                "lora_dropout": 0.0,
                "bias": "none",
                "inference_mode": True,
            },
        )
        return SamplerPublication(requested_version, self.base_model, state.step)

    def publish_sampler_snapshot(self, capture_id):
        model_id, version, weights, config = self.sampler_captures.pop(capture_id)
        path = Path(tempfile.mkdtemp(prefix="spindle-torchtitan-sampler-"))
        try:
            save_file(weights, path / "adapter_model.safetensors")
            (path / "adapter_config.json").write_text(json.dumps(config))
            volume_name = os.environ.get("SPINDLE_BULLETIN_VOLUME")
            bulletin = SnapshotBulletin(
                os.environ["SPINDLE_BULLETIN_ROOT"],
                commit=(lambda: modal.Volume.from_name(volume_name).commit())
                if volume_name
                else None,
            )
            bulletin.publish(VersionRef(model_id, version), path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _volume(self, operation):
        name = os.environ.get("SPINDLE_CHECKPOINT_VOLUME")
        if name:
            volume = modal.Volume.from_name(name)
            if operation == "reload":
                volume.reload()
            else:
                volume.commit()

    def unload_model(self, model_id):
        state = self.jobs.pop(model_id, None)
        if state is not None:
            for module in self.projections.values():
                module.remove_adapter(state.slot)

    def close(self):
        for model_id in list(self.jobs):
            self.unload_model(model_id)
        self.captures.clear()
        self.sampler_captures.clear()


def build_executor():
    # Native TorchTitan imports belong to the GPU worker, not the control plane.
    settings = json.loads(os.environ["SPINDLE_BACKEND_CONFIG"])
    config = TorchTitanConfig(**settings["torchtitan"])
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, adapter_batch, projections = build_model(config, device)
    return DistributedExecutor(
        TorchTitanBackend(
            config,
            base_model=os.environ["SPINDLE_BASE_MODEL"],
            model=model,
            adapter_batch=adapter_batch,
            projections=projections,
            checkpoint_dir=settings.get("checkpoint_dir", "/checkpoints"),
        )
    )
