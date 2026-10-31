from spindle.configs.nemotron3_super_120b_a12b_lora_64k_b300 import Config as Parent


class Config(Parent):
    name = "nemotron35-super-vl-120b-a12b-lora-64k-b300"
    model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


config = Config()
