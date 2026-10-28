"""Mixed-precision sampler fleet sharing a Miles multi-LoRA trainer."""

from spindle.configs.qwen36_35b_a3b_lora_32k import Config as QwenConfig

from . import config as common

LEARNING_RATE = 1e-5
GROUPS = [
    {
        **group,
        **(
            {"moe_runner_backend": "experimental_sgl_trtllm"}
            if group["quantization"] == "nvfp4_online"
            else {}
        ),
    }
    for group in common.GROUPS
]


class Config(QwenConfig):
    name = "qwen36-mimo-mixed-lora-256k"
    model_revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    max_context_length = 262144
    trainer_gpu = "B200"
    trainer_region = ()
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 786432
    inference_cpu = 16
    inference_memory_mib = 262144
    inference_min_replicas = 1
    inference_max_replicas = 2
    inference_target_concurrency = 16
    inference_startup_timeout_s = 2400
    miles_cfg = {  # noqa: RUF012 - BaseConfig copies class dictionaries per instance
        **QwenConfig.miles_cfg,
        "context_parallel_size": 4,
        "max_tokens_per_gpu": 65536,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "model.language_model.layers.*.mlp.shared_expert.gate_proj",
            "model.language_model.layers.*.mlp.shared_expert.up_proj",
            "model.language_model.layers.*.mlp.shared_expert.down_proj",
        ],
    }
    sglang_cfg = {  # noqa: RUF012 - BaseConfig copies class dictionaries per instance
        **QwenConfig.sglang_cfg,
        "sampling_mask_max_tokens": 248320,
        "kv_cache_dtype": "fp8_e4m3",
        "sampling_backend": "pytorch",
    }
    overrides = {  # noqa: RUF012 - BaseConfig copies class dictionaries per instance
        "platform.frontend": "spindle-mimo-mixed-lora",
        "platform.secrets.api": "spindle-mimo-mixed-api-lora",
        "platform.modal.environment": "kailash-dev",
        "platform.modal.region": "us-west",
        "platform.storage.assets": "spindle-mimo-mixed-assets-lora",
        "platform.storage.bulletin": "spindle-mimo-mixed-bulletin-lora",
        "platform.storage.checkpoints": "spindle-mimo-mixed-checkpoints-lora",
    }


config = Config()


def sampler_config(group):
    shared = common.sampler_config(group)
    return Config(
        name=f"{config.name}-{group['name']}",
        inference_gpu=group["gpu"],
        inference_region=shared.inference_region,
        inference_gpus_per_node=group["gpus"],
        inference_env=shared.inference_env,
        overrides={
            "sglang_cfg.tp_size": group["gpus"],
            "sglang_cfg.ep_size": group["gpus"],
            "sglang_cfg.quantization": group["quantization"],
            "sglang_cfg.cpu_weight_cache_max_compile_group_gb": shared.sglang_cfg[
                "cpu_weight_cache_max_compile_group_gb"
            ],
            "sglang_cfg.moe_runner_backend": group.get("moe_runner_backend", "auto"),
        },
    )
