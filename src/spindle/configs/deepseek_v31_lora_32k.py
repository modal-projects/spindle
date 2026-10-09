from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "deepseek-v31-lora-32k"
    model = "deepseek-ai/DeepSeek-V3.1"
    max_context_length = 32768
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_nodes = 2
    trainer_cpu = 32
    trainer_memory_mib = 524288
    trainer_max_clients_per_instance = 2
    miles_cfg = {
        "model_type": "deepseek-v3",
        "tensor_model_parallel_size": 4,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        # kv_b_proj is absorbed into the inference attention kernel.
        "target_modules": [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "max_tokens_per_gpu": 16384,
        "max_lora_slots": 2,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "moe_permute_fusion": False,
            # Multi-LoRA shared-expert collectives must not race expert dispatch.
            "moe_shared_expert_overlap": False,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    inference_gpu = "B300"
    inference_gpus_per_node = 4
    inference_max_replicas = 2
    inference_cpu = 32
    inference_memory_mib = 1048576
    inference_target_concurrency = 48
    inference_startup_timeout_s = 3600
    sglang_cfg = {
        # Cached adapter tensors must not keep the snapshot volume mapped.
        "weight_loader_disable_mmap": True,
        "model_loader_extra_config": {"num_threads": 2},
        "tp_size": 4,
        "ep_size": 4,
        "quantization": "fp8",
        "moe_runner_backend": "triton",
        "mem_fraction_static": 0.9,
        # The four-B300 sampler reports 827K KV tokens with four adapter slots.
        "max_running_requests": 96,
        "max_queued_requests": 32,
        "max_loaded_loras": 4,
        "max_loras_per_batch": 4,
        "disable_shared_experts_fusion": True,
        "lora_strict_loading": True,
    }


config = Config()
