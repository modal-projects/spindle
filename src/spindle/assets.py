"""Transforms applied to downloaded model weights before they become assets.

``dequantize_bf16`` turns a DeepSeek-style quantized release (FP8 dense
weights with 32x32 e8m0 block scales, FP4 routed experts with 1x32 scales)
into a BF16 checkpoint the trainer loads directly. Engram tables stay FP8
with their scales; multi-token-prediction and vision tensors are dropped.
Only NumPy is needed, so the conversion runs in the frontend image.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np

TRANSFORMS = ("dequantize_bf16",)
TRANSFORM_MARKER = ".spindle-transform.json"
_INDEX = "model.safetensors.index.json"
_DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
_QUANTIZED_DTYPES = ("F8_E4M3", "I8")
_FP4_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _fp8_e4m3_table() -> np.ndarray:
    values = np.empty(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            values[byte] = sign * mantissa / 8 * 2.0**-6
        elif exponent == 15 and mantissa == 7:
            values[byte] = math.nan
        else:
            values[byte] = sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return values


_FP8_VALUES = _fp8_e4m3_table()
_E8M0_VALUES = np.array(
    [math.nan if byte == 255 else 2.0 ** (byte - 127) for byte in range(256)],
    dtype=np.float32,
)


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def _read(file, base: int, info: dict) -> np.ndarray:
    start, end = info["data_offsets"]
    file.seek(base + start)
    return np.frombuffer(file.read(end - start), dtype=np.uint8)


def _e8m0(file, base: int, info: dict) -> np.ndarray:
    return _E8M0_VALUES[_read(file, base, info)].reshape(info["shape"])


def to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest even), returned as the raw uint16 bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding) >> 16).astype(np.uint16)


def _scaled(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    (rows, cols), (scale_rows, scale_cols) = values.shape, scales.shape
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(f"scales {scales.shape} do not tile values {values.shape}")
    scales = np.repeat(scales, rows // scale_rows, axis=0)
    scales = np.repeat(scales, cols // scale_cols, axis=1)
    return to_bfloat16(values * scales)


def dequantize_fp8(file, base: int, weight: dict, scale: dict) -> np.ndarray:
    values = _FP8_VALUES[_read(file, base, weight)].reshape(weight["shape"])
    return _scaled(values, _e8m0(file, base, scale))


def dequantize_fp4(file, base: int, weight: dict, scale: dict) -> np.ndarray:
    packed = _read(file, base, weight).reshape(weight["shape"])
    values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    values[:, 0::2] = _FP4_VALUES[packed & 0xF]
    values[:, 1::2] = _FP4_VALUES[packed >> 4]
    return _scaled(values, _e8m0(file, base, scale))


def plan_shard(header: dict) -> dict[str, tuple[str, str, list[int]]]:
    """Map each output tensor to (kind, dtype, shape); kind is copy, fp8 or fp4."""
    outputs: dict[str, tuple[str, str, list[int]]] = {}
    for name, info in header.items():
        if name.startswith(_DROP_PREFIXES) or name.endswith(".scale"):
            continue
        scale = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
        if ".engram.embed." in name:
            outputs[name] = ("copy", info["dtype"], info["shape"])
            outputs[scale] = ("copy", header[scale]["dtype"], header[scale]["shape"])
        elif scale in header and info["dtype"] == "F8_E4M3":
            outputs[name] = ("fp8", "BF16", info["shape"])
        elif scale in header and info["dtype"] == "I8":
            outputs[name] = ("fp4", "BF16", [info["shape"][0], info["shape"][1] * 2])
        elif scale in header:
            raise ValueError(f"unexpected quantized dtype for {name}: {info['dtype']}")
        elif info["dtype"] in _QUANTIZED_DTYPES:
            raise ValueError(f"{name} ({info['dtype']}) has no scale in its shard")
        else:
            outputs[name] = ("copy", info["dtype"], info["shape"])
    return outputs


def convert_shard(source: Path, target: Path) -> dict[str, str]:
    """Write ``target`` with the dequantized tensors of ``source``; returns the
    tensor names it holds, mapped to ``target``'s file name. A ``target`` left by
    an earlier, interrupted conversion is complete and kept."""
    header, base = read_header(source)
    outputs = plan_shard(header)
    if not outputs or target.exists():
        return {name: target.name for name in outputs}
    meta: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (kind, dtype, shape) in outputs.items():
        if kind == "copy":
            start, end = header[name]["data_offsets"]
            size = end - start
        else:
            size = math.prod(shape) * 2
        meta[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(meta, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    partial = target.with_name(f".{target.name}.partial")
    with open(source, "rb") as file, open(partial, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for name, (kind, _, _) in outputs.items():
            if kind == "copy":
                start, end = header[name]["data_offsets"]
                file.seek(base + start)
                remaining = end - start
                while remaining:
                    chunk = file.read(min(remaining, 1 << 30))
                    if not chunk:
                        raise RuntimeError(f"{source} is truncated at {name}")
                    out.write(chunk)
                    remaining -= len(chunk)
            else:
                scale = header[name[: -len(".weight")] + ".scale"]
                convert = dequantize_fp8 if kind == "fp8" else dequantize_fp4
                out.write(convert(file, base, header[name], scale).tobytes())
            if out.tell() - (8 + len(blob)) != meta[name]["data_offsets"][1]:
                raise RuntimeError(f"size mismatch writing {name}")
    partial.replace(target)
    return {name: target.name for name in outputs}


def flatten_config(config: dict) -> dict:
    """The text model's configuration, without the quantization and vision
    sections the BF16 checkpoint no longer carries."""
    config = dict(config)
    text = dict(config.pop("text_config", {}))
    text.pop("model_type", None)
    for key in ("quantization_config", "vision_config", "image_token_id"):
        config.pop(key, None)
    return {**text, **config, "num_nextn_predict_layers": 0, "torch_dtype": "bfloat16"}


def dequantize_checkpoint(source: Path, target: Path) -> None:
    if (target / TRANSFORM_MARKER).exists():
        return
    index = json.loads((source / _INDEX).read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    target.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    for shard in shards:
        weight_map.update(convert_shard(source / shard, target / shard))
    total = sum((target / shard).stat().st_size for shard in set(weight_map.values()))
    (target / _INDEX).write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        ),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(
            flatten_config(json.loads((source / "config.json").read_text())), indent=2
        ),
        encoding="utf-8",
    )
    for path in source.iterdir():
        if path.suffix == ".safetensors" or path.name in (_INDEX, "config.json"):
            continue
        if path.name.startswith("."):
            continue
        if path.is_dir():
            shutil.copytree(path, target / path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target / path.name)
    (target / TRANSFORM_MARKER).write_text(
        json.dumps({"transform": "dequantize_bf16", "tensors": len(weight_map)}),
        encoding="utf-8",
    )


def apply_transform(name: str, source: Path, target: Path) -> None:
    if name == "dequantize_bf16":
        dequantize_checkpoint(source, target)
        return
    raise ValueError(f"unknown asset transform: {name!r}")
