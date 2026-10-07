"""Full-model LoRA checks invoked by glm53_multinode.py."""

import json
import math
import struct
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import modal
import torch
from safetensors import safe_open
from stitch.types import VersionRef

from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_runtime.runtime import MilesRuntime
from spindle.inference.bulletin import SnapshotBulletin


def main():
    torch.manual_seed(42)
    volume = modal.Volume.from_name("spindle-glm53-pr26-full-validation")
    settings = json.loads(Path(sys.argv[1]).read_text())
    settings.update(hf_checkpoint="/validation/model")
    settings["cli_options"].update(global_batch_size=8, micro_batch_size=1)
    output = Path(tempfile.mkdtemp(prefix="check-", dir="/validation"))
    timings = {}
    start = time.monotonic()
    runtime = MilesRuntime(MilesBackendConfig(**settings))
    timings["startup_s"] = time.monotonic() - start
    slots = range(4)
    rank = settings["max_lora_rank"]
    try:
        for slot in slots:
            runtime.load_slot(slot, rank, rank)
        short_rows = None
        for step, length in enumerate((256, 256, 16384)):
            rows = tuple(
                (
                    slot,
                    {
                        "tokens": [10 + slot + i % 100 for i in range(length)],
                        "target_tokens": [
                            10 + slot + i % 100 for i in range(1, length)
                        ],
                        "target_len": length - 1,
                        "weights": [1.0] * (length - 1),
                    },
                )
                for slot in slots
                for _ in range(2 if length == 256 else 1)
            )
            if length == 256:
                short_rows = rows
            start = time.monotonic()
            result = runtime.forward_backward(
                rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=False
            )
            timings[f"step_{step}_forward_backward_s"] = time.monotonic() - start
            assert len(result) == len(rows)
            assert all(
                torch.isfinite(torch.tensor(r["logprobs"])).all() for r in result
            )
            start = time.monotonic()
            metrics = runtime.optim_step(
                {
                    slot: {
                        "learning_rate": 1e-5,
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    }
                    for slot in slots
                }
            )
            timings[f"step_{step}_optimizer_s"] = time.monotonic() - start
            print(
                "UPDATE", step, "sequence_length", length, metrics, timings, flush=True
            )

        start = time.monotonic()
        runtime.export_slot_peft(
            slot=0,
            path=str(output / "adapter"),
            rank=rank,
            alpha=rank,
            base_model="zai-org/GLM-5.3-Flash",
            target_modules=tuple(settings["target_modules"]),
            lora_dropout=0,
        )
        timings["export_s"] = time.monotonic() - start
        adapter_file = output / "adapter/adapter_model.safetensors"
        with adapter_file.open("rb") as handle:
            header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
        counts = Counter()
        for name, tensor in header.items():
            if name != "__metadata__":
                counts[tensor["dtype"]] += math.prod(tensor["shape"])
        print("ADAPTER", adapter_file.stat().st_size, dict(counts), flush=True)

        start = time.monotonic()
        bulletin = SnapshotBulletin("/validation/bulletin", commit=volume.commit)
        ref = VersionRef(output.name, 1)
        bulletin.publish(ref, output / "adapter")
        timings["publish_hash_copy_commit_s"] = time.monotonic() - start
        start = time.monotonic()
        bulletin.resolve(ref)
        timings["resolve_hash_cached_s"] = time.monotonic() - start

        start = time.monotonic()
        runtime.save_slot(0, str(output / "checkpoint"))
        timings["checkpoint_s"] = time.monotonic() - start
        runtime.unload_slot(0)
        start = time.monotonic()
        runtime.load_slot(0, rank, rank, checkpoint=str(output / "checkpoint"))
        timings["restore_s"] = time.monotonic() - start
        runtime.export_slot_peft(
            slot=0,
            path=str(output / "restored"),
            rank=rank,
            alpha=rank,
            base_model="zai-org/GLM-5.3-Flash",
            target_modules=tuple(settings["target_modules"]),
            lora_dropout=0,
        )
        with (
            safe_open(adapter_file, framework="pt") as original,
            safe_open(
                output / "restored/adapter_model.safetensors", framework="pt"
            ) as restored,
        ):
            assert original.keys() == restored.keys()
            for key in original.keys():
                torch.testing.assert_close(
                    original.get_tensor(key),
                    restored.get_tensor(key),
                    rtol=0,
                    atol=0,
                    msg=key,
                )
        result = runtime.forward_backward(
            short_rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=True
        )
        report = {
            "model_path": "/validation/model",
            "adapter_path": str(bulletin.resolve(ref)),
            "tokens": short_rows[0][1]["tokens"][:-1],
            "logprobs": result[0]["logprobs"],
            "timings": timings,
            "adapter_bytes": adapter_file.stat().st_size,
            "adapter_parameters_by_dtype": dict(counts),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2))
        Path("/validation/latest.json").write_text(json.dumps(report, indent=2))
        volume.commit()
        print("PASS", json.dumps(report["timings"]), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
