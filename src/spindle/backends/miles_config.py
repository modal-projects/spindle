from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PEFT_TARGETS = {
    "linear_qkv": ("q_proj", "k_proj", "v_proj"),
    "linear_q": ("q_proj",),
    "linear_k": ("k_proj",),
    "linear_v": ("v_proj",),
    "linear_proj": ("o_proj",),
    "linear_fc1": ("gate_proj", "up_proj"),
    "linear_fc1_gate": ("gate_proj",),
    "linear_fc1_up": ("up_proj",),
    "linear_fc2": ("down_proj",),
    "output_layer": ("lm_head",),
}
_ATTN_LEAVES = frozenset(
    {
        "linear_qkv",
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
)
_MLP_LEAVES = frozenset(
    {
        "linear_fc1",
        "linear_fc1_gate",
        "linear_fc1_up",
        "linear_fc2",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
_UNEMBED_LEAVES = frozenset({"output_layer", "lm_head"})


def lora_target_flags(target_modules: tuple[str, ...]) -> tuple[bool, bool, bool]:
    """Return ``(train_attn, train_mlp, train_unembed)`` implied by target modules."""
    leaves = {module.rsplit(".", 1)[-1] for module in target_modules}
    return (
        bool(leaves & _ATTN_LEAVES),
        bool(leaves & _MLP_LEAVES),
        bool(leaves & _UNEMBED_LEAVES),
    )


@dataclass(frozen=True, slots=True)
class MilesBackendConfig:
    """Stable Spindle configuration translated to Miles CLI arguments at startup."""

    hf_checkpoint: str
    model_type: str
    actor_num_gpus_per_node: int
    actor_num_nodes: int = 1
    megatron_to_hf_mode: str = "bridge"
    tensor_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    max_lora_slots: int = 8
    max_lora_rank: int = 32
    default_lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "output_layer",
    )
    max_tokens_per_gpu: int = 8192
    dynamic_batch_size: bool = True
    align_sequences_to_parallel_layout: bool = False
    extra_args: tuple[str, ...] = ()
    cli_options: dict[str, Any] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.actor_num_nodes * self.actor_num_gpus_per_node

    @property
    def data_parallel_size(self) -> int:
        return self.world_size // (
            self.tensor_model_parallel_size * self.context_parallel_size
        )

    @property
    def sequence_alignment(self) -> int:
        if not self.align_sequences_to_parallel_layout:
            return 1
        return 2 * self.context_parallel_size * self.tensor_model_parallel_size

    @property
    def peft_target_modules(self) -> tuple[str, ...]:
        targets: list[str] = []
        for target in self.target_modules:
            leaf = target.rsplit(".", 1)[-1]
            for name in _PEFT_TARGETS.get(leaf, (leaf,)):
                if name not in targets:
                    targets.append(name)
        return tuple(targets)

    def validate(self) -> None:
        if self.megatron_to_hf_mode not in ("bridge", "raw"):
            raise ValueError("megatron_to_hf_mode must be bridge or raw")
        positive = {
            "actor_num_gpus_per_node": self.actor_num_gpus_per_node,
            "actor_num_nodes": self.actor_num_nodes,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "max_lora_slots": self.max_lora_slots,
            "max_lora_rank": self.max_lora_rank,
            "max_tokens_per_gpu": self.max_tokens_per_gpu,
        }
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.hf_checkpoint:
            raise ValueError("hf_checkpoint is required")
        if not self.model_type and not self.cli_options:
            raise ValueError(
                "model_type or explicit native architecture options are required"
            )
        if (
            not isinstance(self.target_modules, (list, tuple))
            or not self.target_modules
            or not all(isinstance(name, str) and name for name in self.target_modules)
        ):
            raise ValueError("target_modules must be a nonempty list of module names")
        if (
            self.default_lora_alpha <= 0
            or not float(self.default_lora_alpha).is_integer()
        ):
            raise ValueError("default_lora_alpha must be a positive integer")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        model_parallel_size = (
            self.tensor_model_parallel_size * self.context_parallel_size
        )
        if self.world_size % model_parallel_size != 0:
            raise ValueError(
                "actor_num_nodes * actor_num_gpus_per_node must be a multiple of "
                "tensor_model_parallel_size * context_parallel_size"
            )
        if (
            self.actor_num_nodes > 1
            and self.actor_num_gpus_per_node % self.tensor_model_parallel_size != 0
        ):
            raise ValueError(
                "tensor_model_parallel_size must evenly divide "
                "actor_num_gpus_per_node on each node"
            )

    def miles_arguments(self) -> list[str]:
        """Arguments owned by the integration, excluding model architecture flags."""

        arguments = [
            "--train-backend",
            "megatron",
            "--hf-checkpoint",
            self.hf_checkpoint,
            "--load",
            self.hf_checkpoint,
            "--pretrained-checkpoint",
            self.hf_checkpoint,
            "--ref-load",
            self.hf_checkpoint,
            "--megatron-to-hf-mode",
            self.megatron_to_hf_mode,
            "--debug-train-only",
            "--rollout-num-gpus",
            "0",
            "--actor-num-nodes",
            str(self.actor_num_nodes),
            "--actor-num-gpus-per-node",
            str(self.actor_num_gpus_per_node),
            "--multi-lora-n-adapters",
            str(self.max_lora_slots),
            "--lora-rank",
            str(self.max_lora_rank),
            "--lora-alpha",
            str(int(self.default_lora_alpha)),
            "--lora-dropout",
            str(self.lora_dropout),
            "--target-modules",
            ",".join(self.target_modules),
            "--no-gradient-accumulation-fusion",
            "--optimizer",
            "adam",
            "--lr",
            "1e-4",
            "--tensor-model-parallel-size",
            str(self.tensor_model_parallel_size),
            "--pipeline-model-parallel-size",
            "1",
            "--context-parallel-size",
            str(self.context_parallel_size),
            "--expert-model-parallel-size",
            str(self.expert_model_parallel_size),
            "--expert-tensor-parallel-size",
            str(self.expert_tensor_parallel_size),
            *self._batching_arguments(),
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend",
            "flash",
        ]
        if self.tensor_model_parallel_size > 1:
            arguments.append("--sequence-parallel")
        return [*arguments, *self.extra_args]

    def _batching_arguments(self) -> list[str]:
        if self.dynamic_batch_size:
            return [
                "--use-dynamic-batch-size",
                "--max-tokens-per-gpu",
                str(self.max_tokens_per_gpu),
            ]
        # Attention layouts such as bshd take one sequence per micro-batch.
        return ["--micro-batch-size", "1"]


def parse_backend_config(
    value: dict[str, Any],
) -> tuple[MilesBackendConfig, Path, Path]:
    miles = dict(value.get("miles") or value)
    for name in ("target_modules", "extra_args"):
        if name in miles:
            miles[name] = tuple(miles[name])
    config = MilesBackendConfig(**miles)
    config.validate()
    checkpoint_dir = Path(value.get("checkpoint_dir") or "/tmp/spindle")
    capture_dir = Path(value.get("capture_dir") or "/tmp/spindle-miles-captures")
    return config, checkpoint_dir, capture_dir
