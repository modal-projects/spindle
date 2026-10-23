from spindle.configuration import BaseConfig


class Config(BaseConfig):
    """DeepSeek-V4.1-Flash LoRA on attention and the shared expert."""

    name = "deepseek-v41-flash-lora-8k"
    model = "deepseek-ai/DeepSeek-V4.1-Flash"
    # The release is FP8/FP4; training and serving both use exact BF16 weights.
    model_weights = "PrimeIntellect/DeepSeek-V4.1-Flash-bf16"
    weights_revision = "60a77a60c81e63f197c33e0b427c67f26ab27147"
    max_context_length = 8192
    backend = "miles"
    trainer_image = "miles-v0.1.2"
    # Not B200: the trainer deadlocks there in cross-node expert all-to-alls.
    trainer_gpu = ["B300", "H200"]
    trainer_gpus_per_node = 8
    trainer_nodes = 2
    trainer_cpu = 64
    trainer_memory_mib = 1_200_000
    trainer_disk_mib = 600_000
    trainer_timeout_s = 172800
    trainer_max_clients_per_instance = 2
    miles_cfg = {
        "model_type": "deepseek-v4.1",
        "megatron_to_hf_mode": "raw",
        "tensor_model_parallel_size": 4,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        "dynamic_batch_size": False,
        # Whole-path patterns: a bare wkv or wq_b would also select the
        # compressor, indexer or Engram projections of the same name.
        "target_modules": [
            "layers.*.attn.wq_a",
            "layers.*.attn.wkv",
            "layers.*.attn.wq_b",
            "layers.*.attn.wo_b",
            "layers.*.ffn.shared_experts.w1",
            "layers.*.ffn.shared_experts.w2",
            "layers.*.ffn.shared_experts.w3",
        ],
        "max_lora_slots": 2,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            "dsv4_impl": "miles",
            "model_name": "deepseekv41",
            "qkv_format": "bshd",
            "attention_backend": "auto",
            "moe_router_freeze_gate": True,
            "freeze_e_score_correction_bias": True,
            # Multi-LoRA shared-expert collectives must not race expert dispatch.
            "moe_shared_expert_overlap": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "distributed_timeout_minutes": 120,
            "deterministic_mode": True,
        },
    }
    trainer_env = {
        # Lazily loaded kernels can deadlock against in-flight cross-node expert all-to-alls.
        "CUDA_MODULE_LOADING": "EAGER",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TILELANG_CACHE_DIR": "/tmp/tilelang_cache",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    inference_gpu = "B300"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 700_000
    inference_disk_mib = 600_000
    inference_max_replicas = 2
    inference_startup_timeout_s = 7200
    inference_env = {
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
        # A BF16 checkpoint must not be routed to the FP4 expert kernels.
        "SGLANG_DSV4_FP4_EXPERTS": "0",
    }
    sglang_cfg = {
        "tp_size": 8,
        "ep_size": 8,
        "attention_backend": "dsv4",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "mem_fraction_static": 0.7,
        "model_loader_extra_config": {"enable_multithread_load": False},
        "weight_loader_drop_cache_after_load": True,
        "weight_loader_disable_mmap": False,
        "disable_shared_experts_fusion": True,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "max_loaded_loras": 8,
        "max_loras_per_batch": 4,
        "lora_strict_loading": True,
        # Serve the text model only; the checkpoint also carries vision and MTP weights.
        "json_model_override_args": '{"vision_n_layers": 0, "num_nextn_predict_layers": 0}',
    }


config = Config()
