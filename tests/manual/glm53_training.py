"""Four-layer GLM checkpoint check: two slots, updates, restore and PEFT export.

Invoked by validate_glm53.py --train. Uses the upstream validation slice,
preserving full layer widths and expert counts. This integration test does not
load the full 45-layer checkpoint.
"""

import json
import sys
import tempfile
from pathlib import Path

import modal
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

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
    model_config = json.loads((path / "config.json").read_text())["text_config"]
    layers = model_config["num_hidden_layers"]
    dense_layers = model_config["first_k_dense_replace"]
    settings["cli_options"].update(
        num_layers=layers,
        moe_layer_freq=f"[0]*{dense_layers}+[1]*{layers - dense_layers}",
        global_batch_size=2,
        micro_batch_size=1,
    )
    output = Path(tempfile.mkdtemp(prefix="check-", dir="/validation"))
    runtime = MilesRuntime(MilesBackendConfig(**settings))
    try:
        for slot in (0, 1):
            runtime.load_slot(slot, 8, 8)
        rows = tuple(
            (
                slot,
                {
                    "tokens": list(range(10 + slot, 43 + slot)),
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
        repeats = [forward() for _ in range(3)]
        export_args = dict(
            slot=0,
            rank=8,
            alpha=8,
            base_model=str(path),
            target_modules=tuple(settings["target_modules"]),
            lora_dropout=0,
        )
        runtime.export_slot_peft(path=str(output / "adapter"), **export_args)
        runtime.save_slot(0, str(output / "checkpoint"))
        runtime.unload_slot(0)
        runtime.load_slot(0, 8, 8, checkpoint=str(output / "checkpoint"))
        restored = forward()
        runtime.export_slot_peft(path=str(output / "restored"), **export_args)
        original_weights = load_file(str(output / "adapter/adapter_model.safetensors"))
        restored_weights = load_file(str(output / "restored/adapter_model.safetensors"))
        assert original_weights.keys() == restored_weights.keys()
        for key, value in original_weights.items():
            torch.testing.assert_close(
                value, restored_weights[key], rtol=0, atol=0, msg=key
            )
        reference = torch.tensor(after[0]["logprobs"])
        repeat_diff = max(
            (reference - torch.tensor(result[0]["logprobs"])).abs().max().item()
            for result in repeats
        )
        restore_diff = (
            (reference - torch.tensor(restored[0]["logprobs"])).abs().max().item()
        )
        print(
            "RESTORE",
            json.dumps(
                {
                    "adapter_tensors_exact": len(original_weights),
                    "repeated_forward_max_diff": repeat_diff,
                    "restored_forward_max_diff": restore_diff,
                }
            ),
            flush=True,
        )
        report = {
            "model_path": str(path),
            "adapter_path": str(output / "adapter"),
            "tokens": rows[0][1]["tokens"][:-1],
            "logprobs": after[0]["logprobs"],
            "repeat_logprobs": [r[0]["logprobs"] for r in repeats],
            "restored_logprobs": restored[0]["logprobs"],
        }
        Path("/validation/latest.json").write_text(json.dumps(report))
        modal.Volume.from_name("spindle-glm53-pr26-validation").commit()
        print(
            "PASS: two-slot forward/backward, two updates, checkpoint restore, PEFT export",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
