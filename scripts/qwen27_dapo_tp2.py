"""Eight 27B LoRA clients: TP1/DP8 trainer; eight TP2 inference replicas."""
from spindle.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = 'qwen27-dapo-tp2-20261008'
    max_context_length = 32768
    trainer_region = 'us'
    trainer_gpu = 'H200'
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    inference_gpu = 'H200'
    inference_gpus_per_node = 2
    inference_min_replicas = 8
    inference_max_replicas = 8
    frontend_cpu = 4
    frontend_memory_mib = 8192
    sampling_cpu = 4
    sampling_memory_mib = 8192
    sampling_min_containers = 1
    sampling_max_inputs = 32
    session_idle_timeout_s = 86400
    pool_idle_timeout_s = 86400
    overrides = {
        'trainer_env.TORCH_COMPILE_DISABLE': '1',
        'miles_cfg.tensor_model_parallel_size': 1,
        'miles_cfg.max_lora_slots': 8,
        'miles_cfg.max_tokens_per_gpu': 18432,
        'platform.frontend': 'spindle-qwen27-dapo-tp2-20261008',
        'platform.modal.environment': 'lilo-deploy',
        'platform.modal.region': 'us-west',
        'platform.secrets.api': 'spindle-swe-staggered-api',
        'platform.secrets.sampler_proxy': 'lilo-proxy',
        'platform.storage.assets': 'lilo-model-assets',
        'platform.storage.checkpoints': 'qwen27-dapo-checkpoints',
        'platform.storage.bulletin': 'qwen27-dapo-bulletin',
        'sglang_cfg.tp_size': 2,
        'sglang_cfg.max_running_requests': 128,
        'sglang_cfg.max_queued_requests': 256,
    }


config = Config()
