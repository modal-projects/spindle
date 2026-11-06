"""Reversible BF16 XOR publications exercising native quantized staging."""

import json
import struct
from pathlib import Path

import numpy as np
import xxhash
import zstandard
from safetensors.numpy import save_file


def write_validation_lineage(model_path, destination):
    model_path, destination = Path(model_path), Path(destination)
    mapping = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    shard = model_path / mapping[name]
    with shard.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise ValueError("expected BF16 canonical expert weights")
        start, end = entry["data_offsets"]
        source.seek(8 + header_length + start)
        original = np.frombuffer(source.read(end - start), dtype=np.uint16)
    values = (original.astype(np.uint32) << 16).view(np.float32) * np.float32(1.02)
    # Round the modified FP32 tensor to BF16, with ties to even.
    bits = values.view(np.uint32)
    changed = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    delta = np.bitwise_xor(original, changed).view(np.uint8)
    compressed = zstandard.ZstdCompressor(level=1).compress(delta)
    filename = "model-00000-of-00001.safetensors"
    for version, target in ((1, changed), (2, original)):
        directory = destination / f"weight_v{version:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: np.frombuffer(compressed, dtype=np.uint8)},
            str(directory / filename),
            metadata={name: xxhash.xxh3_128_hexdigest(target.view(np.uint8))},
        )
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": f"{version:06d}",
                        "base_version": f"{version - 1:06d}",
                        "delta_encoding": "xor",
                        "compression_format": "zstd",
                        "checksum_format": "xxh3-128",
                    },
                    "weight_map": {name: filename},
                }
            )
        )
    return {
        "tensor": name,
        "shape": entry["shape"],
        "compressed_bytes": len(compressed),
    }


def stage_and_restore(client, model_path, sampling_request):
    source = "/tmp/validation-lineage"
    report = write_validation_lineage(model_path, source)
    # A distinct directory makes SGLang compile the canonical baseline and
    # compare every runtime weight storage against the active startup model.
    # Passing the boot directory would seed from active weights and skip this.
    baseline = Path("/tmp/validation-base-checkpoint")
    baseline.mkdir(exist_ok=True)
    for original in Path(model_path).iterdir():
        (baseline / original.name).symlink_to(
            original, target_is_directory=original.is_dir()
        )
    report["stages"] = []
    report["samples"] = []
    for version in (0, 1, 2):
        staged = client.post(
            "http://127.0.0.1:8001/stage_weight_update",
            json={
                "destination": "cpu",
                "base_checkpoint_dir": str(baseline),
                "checkpoint_source_dir": source,
                "base_version": 0,
                "target_version": version,
            },
            timeout=1800,
        )
        if not staged.is_success:
            raise RuntimeError(f"weight staging failed: {staged.text}")
        report["stages"].append(staged.json())
        if version:
            committed = client.post(
                "http://127.0.0.1:8001/update_weights_from_cpu",
                json={
                    "target_version": version,
                    "flush_cache": True,
                },
                timeout=600,
            )
            if not committed.is_success:
                raise RuntimeError(f"weight commit failed: {committed.text}")
            report["stages"].append(committed.json())
            sampled = client.post(
                "http://127.0.0.1:8001/generate", json=sampling_request, timeout=600
            )
            sampled.raise_for_status()
            report["samples"].append(sampled.json())
    return report
