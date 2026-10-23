"""Check the pinned native Qwen model on a real GPU before large-model tests."""

import modal

from spindle.providers.modal.torchtitan_image import image

app = modal.App("spindle-torchtitan-validation")

with image.imports():
    import torch
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.qwen3_8 import build_model_config


@app.function(image=image, gpu="H200", timeout=1200)
def probe():
    config = build_model_config(
        "debugmodel", parallelism=ParallelismConfig(), seq_len=256
    )
    config.vision_encoder = None
    config.vocab_size = 1024
    config.tok_embeddings.num_embeddings = 1024
    config.lm_head.out_features = 1024
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = config.build()
    model.init_weights(buffer_device=torch.device("cuda"))
    tokens = torch.randint(0, 1024, (256,), device="cuda")
    positions = torch.arange(128, device="cuda").repeat(2)
    metadata = model._get_attention_metadata(positions)
    output = model(tokens, positions=positions, attention_metadata=metadata)
    output.float().square().mean().backward()
    return {
        "torch": torch.__version__,
        "shape": list(output.shape),
        "gpu": torch.cuda.get_device_name(),
    }


@app.local_entrypoint()
def main():
    print(probe.remote())
