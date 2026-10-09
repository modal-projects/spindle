from spindle.configs.nemotron3_super_120b_a12b_lora_64k_b300 import Config as Parent


class Config(Parent):
    name = "nemotron3-super-120b-a12b-lora-256k"
    max_context_length = 262144
    trainer_gpu = "H200"
    trainer_nodes = 2
    trainer_gpus_per_node = 8
    trainer_max_clients_per_instance = 2
    overrides = {
        # Four initialized optimizer slots exhausted memory on a later padded batch.
        "miles_cfg.max_lora_slots": 2,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 8,
        "miles_cfg.expert_model_parallel_size": 16,
        "miles_cfg.align_sequences_to_parallel_layout": True,
        "sglang_cfg.max_loras_per_batch": 2,
    }


config = Config()
