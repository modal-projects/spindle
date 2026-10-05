"""Cache a BF16 serving checkpoint using the trainer's exact FP8 conversion."""

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import modal
import torch
from megatron.bridge.models.glm5_next.glm5_next_bridge import dequant_fp8_blockwise
from safetensors import safe_open
from safetensors.torch import save_file


def main(partition=None):
    torch.set_num_threads(4)
    source = Path("/validation/model")
    target = Path("/validation/model-bf16")
    target.mkdir(exist_ok=True)
    volume = modal.Volume.from_name("spindle-glm53-pr26-full-validation")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    config = json.loads((source / "config.json").read_text())
    quantization = config["quantization_config"]
    assert quantization["quant_method"] == "fp8"
    assert quantization["weight_block_size"] == [128, 128]
    if (target / "conversion.json").exists():
        print("BF16 checkpoint already complete", flush=True)
        return
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print("CONVERT", len(shards), "shards", flush=True)

    def convert(shard):
        output = target / shard
        marker = target / (shard + ".json")
        if marker.exists() and output.exists():
            return json.loads(marker.read_text())
        started = time.monotonic()
        weights = {}
        converted = 0
        with ExitStack() as stack:
            handles = {}

            def get(name):
                filename = weight_map[name]
                if filename not in handles:
                    handles[filename] = stack.enter_context(
                        safe_open(source / filename, framework="pt", device="cpu")
                    )
                return handles[filename].get_tensor(name)

            for name, filename in weight_map.items():
                if filename != shard or name.endswith("_scale_inv"):
                    continue
                weight = get(name)
                if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    assert name + "_scale_inv" in weight_map, name
                    weight = dequant_fp8_blockwise(
                        weight, get(name + "_scale_inv")
                    ).cpu()
                    converted += 1
                weights[name] = weight.contiguous()
            save_file(weights, str(output) + ".tmp", metadata={"format": "pt"})
            os.replace(str(output) + ".tmp", output)
            result = {
                "shard": shard,
                "bytes": sum(t.numel() * t.element_size() for t in weights.values()),
                "converted": converted,
                "seconds": time.monotonic() - started,
            }
        marker.write_text(json.dumps(result))
        print("SHARD", json.dumps(result), flush=True)
        return result

    selected = shards if partition is None else shards[partition::4]
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(convert, selected))
    if partition is not None:
        volume.commit()
        print("PARTITION COMPLETE", partition, flush=True)
        return
    index["weight_map"] = {
        k: v for k, v in weight_map.items() if not k.endswith("_scale_inv")
    }
    index["metadata"]["total_size"] = sum(row["bytes"] for row in results)
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy2(path, target / path.name)
    config.pop("quantization_config")
    config["text_config"].pop("quantization_config", None)
    (target / "config.json").write_text(json.dumps(config, indent=2))
    (target / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (target / "conversion.json").write_text(json.dumps(results, indent=2))
    volume.commit()
    print("PASS BF16", index["metadata"]["total_size"], flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
