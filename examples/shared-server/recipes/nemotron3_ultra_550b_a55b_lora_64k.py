from spindle.configs.nemotron3_ultra_550b_a55b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-nemotron3-ultra-550b-a55b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
