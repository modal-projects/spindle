from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "nemotron3-nano-30b-a3b-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    max_context_length = 65536
    trainer_gpu = "H200"
    trainer_gpus_per_node = 2
    trainer_cpu = 16
    trainer_memory_mib = 131072
    trainer_max_clients_per_instance = 6
    miles_cfg = {
        "model_type": "nemotron-3-nano-30b-a3b",
        "tensor_model_parallel_size": 2,
        "expert_model_parallel_size": 2,
        "expert_tensor_parallel_size": 1,
        # Attention and expert MLP adapters; the Mamba state remains frozen.
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
        ],
        "max_tokens_per_gpu": 65536,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            # Nemotron uses ReLU squared, which the SwiGLU fusion does not support.
            "bias_swiglu_fusion": False,
            "moe_token_dispatcher_type": "alltoall",
            "moe_permute_fusion": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    inference_gpu = "H200"
    inference_gpus_per_node = 1
    inference_memory_mib = 131072
    inference_max_replicas = 4
    inference_target_concurrency = 64
    sglang_cfg = {
        "tp_size": 1,
        "dtype": "bfloat16",
        # FlashInfer CUTLASS is the default for this model but cannot apply MoE LoRA.
        "moe_runner_backend": "triton",
        "mem_fraction_static": 0.8,
        "max_running_requests": 128,
        "max_queued_requests": 32,
        "max_loaded_loras": 8,
        "max_loras_per_batch": 6,
        "lora_strict_loading": True,
        "schedule_policy": "lpm",
        "trust_remote_code": True,
    }


config = Config()
