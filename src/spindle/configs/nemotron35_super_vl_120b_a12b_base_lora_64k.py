from spindle.configs.nemotron35_super_vl_120b_a12b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-super-vl-120b-a12b-base-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16"


config = Config()
