"""Four-layer GLM checkpoint check: two slots, updates, restore and PEFT export.

Invoked by validate_glm53.py --train. Uses the upstream validation slice, preserving full layer widths and expert
counts. This is an integration test; the full 45-layer checkpoint is not loaded.
"""

import json
import sys
from pathlib import Path

import modal
import torch
from huggingface_hub import snapshot_download

from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_runtime.runtime import MilesRuntime


def main():
    torch.manual_seed(42)
    path = Path(
        snapshot_download(
            "CharyZeng/GLM-5.3-Flash-4layer", local_dir="/validation/model"
        )
    )
    modal.Volume.from_name("spindle-glm53-pr26-validation").commit()
    print("Downloaded four-layer GLM checkpoint", flush=True)
    settings = json.loads(Path(sys.argv[1]).read_text())
    settings.update(
        hf_checkpoint=str(path),
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        max_tokens_per_gpu=256,
        max_lora_slots=2,
        max_lora_rank=8,
        default_lora_alpha=8,
        extra_args=["--seq-length", "256"],
    )
    settings["cli_options"].update(
        num_layers=4,
        moe_layer_freq="[0]*3+[1]",
        global_batch_size=2,
        micro_batch_size=1,
    )
    runtime = MilesRuntime(MilesBackendConfig(**settings))
    try:
        for slot in (0, 1):
            runtime.load_slot(slot, 8, 8)
        rows = tuple(
            (
                slot,
                {
                    "tokens": list(range(10 + slot, 42 + slot)),
                    "target_tokens": list(range(11 + slot, 43 + slot)),
                    "target_len": 32,
                    "weights": [1.0] * 32,
                },
            )
            for slot in (0, 1)
        )

        def forward():
            return runtime.forward_backward(
                rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=True
            )

        before = forward()
        for step in range(2):
            out = runtime.forward_backward(
                rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=False
            )
            assert all(
                torch.isfinite(torch.tensor(row["logprobs"])).all() for row in out
            )
            metrics = runtime.optim_step(
                {
                    slot: {
                        "learning_rate": 1e-4,
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    }
                    for slot in (0, 1)
                }
            )
            print("UPDATE", step, metrics, flush=True)
        after = forward()
        assert any(a["logprobs"] != b["logprobs"] for a, b in zip(before, after))
        runtime.save_slot(0, "/tmp/glm53-test-checkpoint")
        runtime.unload_slot(0)
        runtime.load_slot(0, 8, 8, checkpoint="/tmp/glm53-test-checkpoint")
        restored = forward()
        torch.testing.assert_close(
            torch.tensor(after[0]["logprobs"]),
            torch.tensor(restored[0]["logprobs"]),
            rtol=0,
            atol=1e-5,
        )
        runtime.export_slot_peft(
            slot=0,
            path="/tmp/glm53-test-adapter",
            rank=8,
            alpha=8,
            base_model=str(path),
            target_modules=tuple(settings["target_modules"]),
            lora_dropout=0,
        )
        assert Path("/tmp/glm53-test-adapter/adapter_model.safetensors").is_file()
        print(
            "PASS: two-slot forward/backward, two updates, checkpoint restore, PEFT export",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
