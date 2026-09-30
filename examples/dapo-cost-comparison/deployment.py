"""Eight H200s with TP=1/DP=8; eight resident LoRA clients."""
from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = 'dapo-instruct-tp1-dp8-20260930'
    trainer_region = 'us'
    trainer_gpu = 'H200'
    trainer_gpus_per_node = 8
    max_context_length = 32768
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    inference_min_replicas = 8
    inference_max_replicas = 8
    session_idle_timeout_s = 86400
    pool_idle_timeout_s = 86400
    overrides = {
        'miles_cfg.tensor_model_parallel_size': 1,
        'miles_cfg.max_lora_slots': 8,
        'miles_cfg.max_tokens_per_gpu': 32768,
        'platform.frontend': 'spindle-dapo-comparison-20260930',
        'platform.modal.environment': 'lilo-deploy',
        'platform.modal.region': 'us-west',
        'platform.secrets.api': 'spindle-swe-staggered-api',
        'platform.secrets.sampler_proxy': 'lilo-proxy',
        'platform.storage.assets': 'lilo-model-assets',
        'platform.storage.checkpoints': 'spindle-dapo-comparison-checkpoints',
        'platform.storage.bulletin': 'spindle-dapo-comparison-bulletin',
        'sglang_cfg.max_running_requests': 64,
        'sglang_cfg.max_queued_requests': 128,
    }


config = Config()
