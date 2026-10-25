"""Article settings and explicitly chosen values for its unspecified details."""

from spindle.configs.qwen36_35b_a3b_fft_64k import Config as QwenConfig


class Config(QwenConfig):
    name = "qwen36-mimo-mixed-256k"
    model_revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    max_context_length = 262144
    trainer_gpu = "B200"
    trainer_region = ()  # Allow any region while the eight-B200 pool is constrained.
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 786432
    inference_cpu = 16
    inference_memory_mib = 262144
    inference_min_replicas = 1  # Keep early groups warm while the rest initialize.
    inference_max_replicas = 2
    inference_target_concurrency = 16
    inference_startup_timeout_s = 2400
    overrides = {  # noqa: RUF012 - BaseConfig copies class dictionaries per instance
        "platform.frontend": "spindle-mimo-mixed",
        "platform.secrets.api": "spindle-mimo-mixed-api",
        "platform.modal.environment": "kailash-dev",
        "platform.modal.region": "us-west",
        "platform.storage.assets": "spindle-mimo-mixed-assets",
        "platform.storage.bulletin": "spindle-mimo-mixed-bulletin",
        "platform.storage.checkpoints": "spindle-mimo-mixed-checkpoints",
        "megatron_cfg.max_tokens_per_microbatch": 262144,
        "megatron_cfg.tensor_model_parallel_size": 2,
        "megatron_cfg.context_parallel_size": 4,
        "megatron_cfg.expert_model_parallel_size": 8,
        "megatron_cfg.provider_overrides.recompute_granularity": "full",
        "megatron_cfg.provider_overrides.recompute_method": "uniform",
        "megatron_cfg.provider_overrides.recompute_num_layers": 1,
        "megatron_cfg.provider_overrides.moe_layer_recompute": False,
        "megatron_cfg.optimizer.lr": 1e-6,
        "megatron_cfg.optimizer.min_lr": 1e-6,
        "sglang_cfg.dp_size": 1,
        "sglang_cfg.enable_dp_attention": False,
        "sglang_cfg.kv_cache_dtype": "fp8_e4m3",
        "sglang_cfg.sampling_mask_max_tokens": 248320,
        "sglang_cfg.mem_fraction_static": 0.75,
        "sglang_cfg.sampling_backend": "pytorch",
    }


# Dataset/model, 256K context, 128 x 8 batch, asynchronous policy updates,
# accelerator/precision pairs, and FP8 KV cache follow the article.
DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATASET_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
GROUPS = [
    {
        "name": "b200-nvfp4",
        "gpu": "B200",
        "gpus": 2,
        "quantization": "nvfp4_online",
        "moe_runner_backend": "flashinfer_trtllm",
    },
    {
        "name": "b300-nvfp4",
        "gpu": "B300",
        "gpus": 2,
        "quantization": "nvfp4_online",
        "moe_runner_backend": "flashinfer_trtllm",
    },
    {"name": "h100-fp8", "gpu": "H100!", "gpus": 4, "quantization": "fp8"},
    {"name": "h100-bf16", "gpu": "H100!", "gpus": 4, "quantization": None},
    {"name": "h200-fp8", "gpu": "H200", "gpus": 2, "quantization": "fp8"},
    {"name": "h200-bf16", "gpu": "H200", "gpus": 2, "quantization": None},
    {"name": "a100-bf16", "gpu": "A100-80GB", "gpus": 4, "quantization": None},
    {"name": "rtx6000-bf16", "gpu": "RTX-PRO-6000", "gpus": 4, "quantization": None},
]

# The article omits these settings; they are reproduction assumptions.
LEARNING_RATE = 1e-6
TEMPERATURE = 1.0
TOP_P = 0.9
MIS_LOW = 0.2
MIS_HIGH = 5.0
PROMPTS_PER_UPDATE = 128
TRAJECTORIES_PER_PROMPT = 8
MAX_TURNS = 200
MAX_TOKENS_PER_TURN = 262144  # Each call is limited by its remaining context budget.
TRAINING_STEPS = 200
CHECKPOINT_INTERVAL = 10
SEED = 42

config = Config()


def sampler_config(group):
    return Config(
        name=f"{config.name}-{group['name']}",
        inference_gpu=group["gpu"],
        inference_gpus_per_node=group["gpus"],
        inference_env={
            "SGLANG_FP8_BLOCK_ONLINE": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            # Preserve Qwen's recurrent gates and routers in their source dtype.
            # These are also excluded by Qwen's official FP8 configuration.
            "SGLANG_FP8_IGNORED_LAYERS": (
                "in_proj_a,in_proj_b,in_proj_ba,mlp.gate,shared_expert_gate,visual,lm_head"
            ),
        }
        if group["quantization"] == "fp8"
        else {},
        overrides={
            "sglang_cfg.tp_size": group["gpus"],
            "sglang_cfg.ep_size": group["gpus"],
            "sglang_cfg.quantization": group["quantization"],
            "sglang_cfg.cpu_weight_cache_max_compile_group_gb": 2
            if group["quantization"] == "fp8"
            else 32,
            "sglang_cfg.moe_runner_backend": group.get("moe_runner_backend", "auto"),
        },
    )
