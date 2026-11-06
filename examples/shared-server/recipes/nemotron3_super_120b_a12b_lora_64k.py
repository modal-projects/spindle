from spindle.configs.nemotron3_super_120b_a12b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-nemotron3-super-120b-a12b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
