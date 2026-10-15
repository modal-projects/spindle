from spindle.configs.nemotron3_super_120b_a12b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron3-super-120b-a12b-lora-64k-b300"
    trainer_gpu = "B300"
    trainer_gpus_per_node = 4
    inference_gpu = "B300"
    inference_gpus_per_node = 2
    overrides = {
        "miles_cfg.expert_model_parallel_size": 4,
        "sglang_cfg.tp_size": 2,
        "sglang_cfg.ep_size": 2,
    }


config = Config()
