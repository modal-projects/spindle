from spindle.configuration import BaseConfig


class Config(BaseConfig):
    """DeepSeek-V4.1-Flash full-parameter training on two B300 nodes.

    The FP8/FP4 release is converted to BF16 while preparing assets; Adam
    state streams to local disk between steps because it does not fit on
    16 GPUs. Samplers keep the Engram tables in host memory and stage weight
    updates on local disk.
    """

    name = "deepseek-v41-flash-fft-8k"
    parameterization = "full"
    model = "deepseek-ai/DeepSeek-V4.1-Flash"
    asset_transform = "dequantize_bf16"
    max_context_length = 8192
    backend = "miles"
    trainer_gpu = "B300"
    trainer_gpus_per_node = 8
    trainer_nodes = 2
    trainer_cpu = 64
    trainer_memory_mib = 1_200_000
    trainer_disk_mib = 3_000_000
    trainer_timeout_s = 172800
    miles_cfg = {
        "model_type": "deepseek-v4.1",
        "megatron_to_hf_mode": "raw",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "expert_model_parallel_size": 8,
        "dynamic_batch_size": False,
        "cli_options": {
            "dsv4_impl": "miles",
            "model_name": "deepseekv41",
            "qkv_format": "bshd",
            "attention_backend": "dsv4",
            "moe_router_freeze_gate": True,
            "freeze_e_score_correction_bias": True,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "grad_reduce_in_bf16": True,
            "accumulate_allreduce_grads_in_fp32": False,
            "stream_optimizer_state_to_disk": True,
            "stream_optimizer_state_moment_dtype": "bf16",
            "offload_train_disk_dir": "/tmp/spindle-optimizer-state",
            "distributed_timeout_minutes": 120,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TILELANG_CACHE_DIR": "/tmp/tilelang_cache",
    }
    sampler_persistence_concurrency = 1
    inference_gpu = "B300"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 700_000
    inference_disk_mib = 2_500_000
    inference_max_replicas = 2
    inference_startup_timeout_s = 3600
    inference_env = {
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
    }
    sglang_cfg = {
        "tp_size": 8,
        "ep_size": 8,
        "attention_backend": "dsv4",
        "trust_remote_code": True,
        "mem_fraction_static": 0.7,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "weight_update_staging": "disk",
        "weight_update_local_checkpoint_dir": "/tmp/spindle-canonical-checkpoint",
    }


config = Config()
