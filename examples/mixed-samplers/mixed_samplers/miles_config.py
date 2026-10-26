"""Multi-LoRA Miles version of the mixed-sampler recipe."""

from spindle.configs.qwen36_35b_a3b_lora_32k import Config as QwenLoraConfig


class Config(QwenLoraConfig):
    name = "qwen36-mimo-miles-256k"
    model_revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    max_context_length = 262144
    trainer_gpu = "B200"
    trainer_region = ()
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 786432
    trainer_max_clients_per_instance = 2
    inference_cpu = 16
    inference_memory_mib = 262144
    inference_min_replicas = 1
    inference_max_replicas = 2
    inference_target_concurrency = 16
    inference_startup_timeout_s = 3600
    overrides = {  # noqa: RUF012 - BaseConfig copies dictionaries per instance
        "platform.frontend": "spindle-mimo-miles-mixed",
        "platform.secrets.api": "spindle-mimo-mixed-api",
        "platform.modal.environment": "kailash-dev",
        "platform.modal.region": "us-west",
        "platform.storage.assets": "spindle-mimo-mixed-assets",
        "platform.storage.bulletin": "spindle-mimo-miles-bulletin",
        "platform.storage.checkpoints": "spindle-mimo-miles-checkpoints",
        "miles_cfg.max_lora_slots": 2,
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.context_parallel_size": 4,
        "miles_cfg.expert_model_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.align_sequences_to_parallel_layout": True,
        "miles_cfg.cli_options.moe_aux_loss_coeff": [0.0],
        "sglang_cfg.dp_size": 1,
        "sglang_cfg.enable_dp_attention": False,
        "sglang_cfg.kv_cache_dtype": "fp8_e4m3",
        "sglang_cfg.sampling_mask_max_tokens": 248320,
        "sglang_cfg.mem_fraction_static": 0.75,
        "sglang_cfg.sampling_backend": "pytorch",
        "sglang_cfg.max_loaded_loras": 8,
        "sglang_cfg.max_loras_per_batch": 2,
    }


config = Config()


def sampler_config(group):
    # The frozen base is quantized once; each client publishes its BF16 adapter.
    # Hopper/Ampere use the Triton LoRA MoE path (TP, EP=1). Blackwell NVFP4
    # requires SGLang's LoRA-capable TRTLLM runner.
    return Config(
        name=f"{config.name}-{group['name']}",
        inference_gpu=group["gpu"],
        inference_region=()
        if group["name"] in ("a100-bf16", "b300-nvfp4")
        else None,
        inference_gpus_per_node=group["gpus"],
        inference_env={
            "SGLANG_EXPERIMENTAL_LORA_OPTI": "1"
            if group["quantization"] == "nvfp4_online"
            else "0",
            # Avoid the experimental cross-stream LoRA path that faulted on
            # B300 under mixed prefill/decode load. This changes kernel
            # scheduling only; adapters and asynchronous RL are unchanged.
            **(
                {"SGLANG_TWO_STREAM_MAX_TOKENS": "0"}
                if group["quantization"] == "nvfp4_online"
                else {}
            ),
            "SGLANG_FP8_BLOCK_ONLINE": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            "SGLANG_FP8_IGNORED_LAYERS": "in_proj_a,in_proj_b,in_proj_ba,mlp.gate,shared_expert_gate,visual,lm_head",
        },
        overrides={
            "sglang_cfg.tp_size": group["gpus"],
            "sglang_cfg.ep_size": 1,
            "sglang_cfg.quantization": group["quantization"],
            "sglang_cfg.lora_backend": "triton"
            if group["quantization"] == "nvfp4_online"
            else "csgmv",
            "sglang_cfg.lora_use_virtual_experts": group["quantization"]
            == "nvfp4_online",
            "sglang_cfg.moe_runner_backend": "experimental_sgl_trtllm"
            if group["quantization"] == "nvfp4_online"
            else "triton",
        },
    )
