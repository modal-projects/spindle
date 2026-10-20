import json
import struct

import numpy as np
import pytest

from spindle import assets


def _write_shard(path, tensors):
    meta, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        meta[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header = json.dumps(meta).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(blobs))


def _read_shard(path):
    header, base = assets.read_header(path)
    raw = path.read_bytes()
    return {
        name: (
            info["dtype"],
            np.frombuffer(
                raw[base + info["data_offsets"][0] : base + info["data_offsets"][1]],
                dtype=np.uint8,
            ),
        )
        for name, info in header.items()
    }


def _bf16_to_float(bits: np.ndarray) -> np.ndarray:
    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _fp8(byte: int) -> float:
    return float(assets._FP8_VALUES[byte])


def test_bfloat16_rounds_to_nearest_even() -> None:
    values = np.array([1.0, 1.00390625, 1.01171875, -3.5e38], dtype=np.float32)
    rounded = _bf16_to_float(assets.to_bfloat16(values))
    assert rounded[0] == 1.0
    assert rounded[1] == 1.0  # halfway, rounds to even mantissa
    assert rounded[2] == pytest.approx(1.015625)
    assert rounded[3] == pytest.approx(-3.5e38, rel=1e-2)


def test_dequantize_checkpoint_converts_each_tensor_kind(tmp_path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 126, size=(64, 96), dtype=np.uint8)
    dense_scale = rng.integers(120, 130, size=(2, 3), dtype=np.uint8)
    codes = rng.integers(0, 16, size=(8, 64), dtype=np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    expert_scale = rng.integers(120, 130, size=(8, 2), dtype=np.uint8)
    engram = rng.integers(0, 255, size=(5, 64), dtype=np.uint8)
    engram_scale = rng.integers(120, 130, size=(5, 2), dtype=np.uint8)
    gate = np.arange(8, dtype=np.float32)
    _write_shard(
        source / "model-00001-of-00002.safetensors",
        {
            "layers.0.attn.wq_a.weight": ("F8_E4M3", dense),
            "layers.0.attn.wq_a.scale": ("F8_E8M0", dense_scale),
            "layers.0.ffn.experts.0.w1.weight": ("I8", packed),
            "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", expert_scale),
            "layers.1.engram.embed.weight": ("F8_E4M3", engram),
            "layers.1.engram.embed.scale": ("F8_E8M0", engram_scale),
            "layers.0.ffn.gate.bias": ("F32", gate),
            "mtp.0.attn.wq_a.weight": ("F8_E4M3", dense),
        },
    )
    _write_shard(
        source / "model-00002-of-00002.safetensors",
        {"vision.blocks.0.attn.wo.weight": ("BF16", np.zeros(4, dtype=np.uint16))},
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.attn.wq_a.weight": "model-00001-of-00002.safetensors",
                    "vision.blocks.0.attn.wo.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "quantization_config": {"quant_method": "fp8"},
                "text_config": {"model_type": "deepseek_v41_text", "hidden_size": 8},
                "vision_config": {"hidden_size": 4},
                "image_token_id": 7,
            }
        )
    )
    (source / "tokenizer.json").write_text("{}")

    target = tmp_path / "bf16"
    assets.dequantize_checkpoint(source, target)

    tensors = _read_shard(target / "model-00001-of-00002.safetensors")
    assert set(tensors) == {
        "layers.0.attn.wq_a.weight",
        "layers.0.ffn.experts.0.w1.weight",
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.0.ffn.gate.bias",
    }
    assert not (target / "model-00002-of-00002.safetensors").exists()

    dtype, bits = tensors["layers.0.attn.wq_a.weight"]
    got = _bf16_to_float(bits).reshape(64, 96)
    expected = np.array(
        [
            [
                _fp8(dense[i, j]) * 2.0 ** (int(dense_scale[i // 32, j // 32]) - 127)
                for j in range(96)
            ]
            for i in range(64)
        ],
        dtype=np.float32,
    )
    assert dtype == "BF16"
    np.testing.assert_array_equal(got, _bf16_to_float(assets.to_bfloat16(expected)))

    dtype, bits = tensors["layers.0.ffn.experts.0.w1.weight"]
    got = _bf16_to_float(bits).reshape(8, 64)
    expected = np.array(
        [
            [
                float(assets._FP4_VALUES[codes[i, j]])
                * 2.0 ** (int(expert_scale[i, j // 32]) - 127)
                for j in range(64)
            ]
            for i in range(8)
        ],
        dtype=np.float32,
    )
    assert dtype == "BF16"
    np.testing.assert_array_equal(got, _bf16_to_float(assets.to_bfloat16(expected)))

    assert tensors["layers.1.engram.embed.weight"][0] == "F8_E4M3"
    np.testing.assert_array_equal(
        tensors["layers.1.engram.embed.weight"][1], engram.ravel()
    )
    assert tensors["layers.1.engram.embed.scale"][0] == "F8_E8M0"
    assert tensors["layers.0.ffn.gate.bias"][0] == "F32"

    index = json.loads((target / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    config = json.loads((target / "config.json").read_text())
    assert config["hidden_size"] == 8
    assert config["torch_dtype"] == "bfloat16"
    assert config["num_nextn_predict_layers"] == 0
    assert "quantization_config" not in config and "vision_config" not in config
    assert (target / "tokenizer.json").exists()

    marker = (target / assets.TRANSFORM_MARKER).read_text()
    (target / "model-00001-of-00002.safetensors").unlink()
    assets.dequantize_checkpoint(source, target)  # completed conversions are not redone
    assert not (target / "model-00001-of-00002.safetensors").exists()
    assert (target / assets.TRANSFORM_MARKER).read_text() == marker


def test_quantized_tensor_without_scale_is_rejected(tmp_path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_shard(
        shard, {"layers.0.attn.wq_a.weight": ("F8_E4M3", np.zeros((4, 4), np.uint8))}
    )

    with pytest.raises(ValueError, match="no scale"):
        assets.convert_shard(shard, tmp_path / "out.safetensors")


def test_converted_shard_is_kept_on_retry(tmp_path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {"norm.weight": ("F32", np.ones(4, np.float32))})
    target = tmp_path / "out.safetensors"
    target.write_bytes(b"done")

    assert assets.convert_shard(shard, target) == {"norm.weight": target.name}
    assert target.read_bytes() == b"done"
