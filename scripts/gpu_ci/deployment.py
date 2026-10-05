"""Same workload topology as the 8-client DAPO throughput benchmark."""

from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    trainer_timeout_s = 10800
    max_context_length = 32768
    inference_gpu = "H200"
    inference_gpus_per_node = 2
    inference_min_replicas = 8
    inference_max_replicas = 8
    session_idle_timeout_s = 3600
    pool_idle_timeout_s = 3600
    overrides = {
        "platform.modal.region": "us",
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 8,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "sglang_cfg.tp_size": 2,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 256,
    }
