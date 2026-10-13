from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen35-4b-miles-fft-64k"
    parameterization = "full"
    model = "Qwen/Qwen3.5-4B"
    max_context_length = 65536
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_memory_mib = 131072
    backend = "miles"
    miles_cfg = {
        "model_type": "qwen3.5-4B",
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 2,
        "max_tokens_per_gpu": 65536,
        "cli_options": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    sampler_persistence_concurrency = 1
    inference_gpu = "H100"
    sglang_cfg = {
        "tp_size": 1,
        "mem_fraction_static": 0.85,
        "max_running_requests": 32,
        "max_queued_requests": 4,
        "weight_update_max_compile_group_gb": 16,
    }


config = Config()
