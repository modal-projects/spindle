"""Packed-backend correctness and throughput; not asynchronous SDK latency."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

from huggingface_hub import snapshot_download
import torch
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData

from spindle.backends.contract import ForwardBatch, ForwardItem, ModelSpec
from spindle.backends.torchtitan_config import TorchTitanConfig
from spindle.backends.torchtitan_lora import TorchTitanBackend
from spindle.backends.torchtitan_runtime.model import build_model
from spindle.providers.modal.torchtitan_image import TORCHTITAN_COMMIT


def make_data(index, sequence_length):
    generator = torch.Generator().manual_seed(100 + index)
    tokens = torch.randint(
        100, 2000, (sequence_length + 1,), generator=generator
    ).tolist()
    return Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=tokens[1:], dtype="int64", shape=[sequence_length]
            ),
            "weights": TensorData(
                data=[1.0 / sequence_length] * sequence_length,
                dtype="float32",
                shape=[sequence_length],
            ),
        },
    )


def check_packed_gradients(backend, items):
    backend.forward_backward(ForwardBatch(items, "cross_entropy"))
    expected = {
        item.model_id: [
            p.grad.detach().cpu()
            for p in backend.parameters_for(backend.jobs[item.model_id].slot)
        ]
        for item in items
    }
    for state in backend.jobs.values():
        state.optimizer.zero_grad(set_to_none=True)
        state.accumulating = False
    errors = {}
    for item in items:
        backend.forward_backward(ForwardBatch((item,), "cross_entropy"))
        state = backend.jobs[item.model_id]
        difference = reference = 0.0
        for parameter, packed_gradient in zip(
            backend.parameters_for(state.slot), expected.pop(item.model_id), strict=True
        ):
            serial_gradient = parameter.grad.detach().cpu()
            difference += (serial_gradient - packed_gradient).square().sum().item()
            reference += serial_gradient.square().sum().item()
        relative_error = (difference / max(reference, 1e-30)) ** 0.5
        assert relative_error < 0.025, (
            f"packed gradients differ for {item.model_id}: {relative_error}"
        )
        errors[item.model_id] = relative_error
        state.optimizer.zero_grad(set_to_none=True)
        state.accumulating = False
    return errors


def run(model_name, flavor, sequence_length, adapter_counts, steps, compile_layers):
    print(f"Downloading {model_name}", flush=True)
    cached = Path("/assets") / model_name
    if not (cached / "config.json").exists():
        cached = Path(
            snapshot_download(model_name, allow_patterns=["*.json", "*.safetensors"])
        )
    config = TorchTitanConfig(
        hf_checkpoint=str(cached),
        model_family="qwen3_8" if "3.8" in model_name else "qwen3_5",
        model_flavor=flavor,
        max_context_length=sequence_length,
        max_lora_slots=max(adapter_counts),
        max_lora_rank=32,
        max_batch_rows=max(adapter_counts),
        max_batch_tokens=sequence_length * max(adapter_counts),
        compile_layers=compile_layers,
    )
    device = torch.device("cuda", 0)
    print(f"Building {model_name} on {torch.cuda.get_device_name()}", flush=True)
    model, routing, projections = build_model(config, device)
    print("Model loaded", flush=True)
    backend = TorchTitanBackend(
        config,
        base_model=model_name,
        model=model,
        adapter_batch=routing,
        projections=projections,
        checkpoint_dir=tempfile.mkdtemp(),
    )
    adam = AdamParams(learning_rate=1e-4)
    results = []
    try:
        for count in adapter_counts:
            print(f"Validating {count} adapters", flush=True)
            ids = tuple(f"adapter-{i}" for i in range(count))
            for i, model_id in enumerate(ids):
                backend.accept_model(
                    model_id,
                    ModelSpec(
                        model_name,
                        "lora",
                        LoraConfig(
                            rank=16,
                            seed=i,
                            train_attn=True,
                            train_mlp=True,
                            train_unembed=False,
                        ),
                    ),
                )
            items = tuple(
                ForwardItem(model_id, (make_data(i, sequence_length),))
                for i, model_id in enumerate(ids)
            )
            batch = ForwardBatch(items, "cross_entropy")
            for _ in range(2):
                backend.forward_backward(batch)
                backend.optim_step(ids, adam)
            # Compare a nonzero trained adapter, including document isolation.
            packed = backend.forward_backward(
                ForwardBatch(items, "cross_entropy", forward_only=True)
            )
            for item, output in zip(items, packed, strict=True):
                (serial,) = backend.forward_backward(
                    ForwardBatch((item,), "cross_entropy", forward_only=True)
                )
                torch.testing.assert_close(
                    torch.tensor(output.loss_fn_outputs[0]["logprobs"].data),
                    torch.tensor(serial.loss_fn_outputs[0]["logprobs"].data),
                    atol=0.05,
                    rtol=0.005,
                )
            gradient_errors = check_packed_gradients(backend, items)
            print(f"Measuring {count} adapters", flush=True)
            # A publication must survive a subsequent optimizer update unchanged.
            receipt = backend.capture_sampler_snapshot(ids[0], "validation-export", 1)
            saved = {
                key: value.clone()
                for key, value in backend.sampler_captures["validation-export"][
                    2
                ].items()
            }
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(steps):
                backend.forward_backward(batch)
                outcomes = backend.optim_step(ids, adam)
                assert all(o.metrics["update_successful:mean"] == 1 for o in outcomes)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            packed_peak_allocated = torch.cuda.max_memory_allocated()
            packed_peak_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.synchronize()
            serial_started = time.perf_counter()
            for _ in range(steps):
                for item in items:
                    backend.forward_backward(ForwardBatch((item,), "cross_entropy"))
                    (outcome,) = backend.optim_step((item.model_id,), adam)
                    assert outcome.metrics["update_successful:mean"] == 1
            torch.cuda.synchronize()
            serial_elapsed = time.perf_counter() - serial_started
            for key, value in backend.sampler_captures.pop("validation-export")[
                2
            ].items():
                torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
            result = {
                "adapters": count,
                "steps": steps,
                "tokens_per_step": count * sequence_length,
                "elapsed_s": elapsed,
                "step_s": elapsed / steps,
                "tokens_per_s": count * sequence_length * steps / elapsed,
                "serial_step_s": serial_elapsed / steps,
                "serial_tokens_per_s": count * sequence_length * steps / serial_elapsed,
                "speedup_over_serial": serial_elapsed / elapsed,
                "packed_gradient_relative_errors": gradient_errors,
                "peak_allocated_bytes": packed_peak_allocated,
                "peak_reserved_bytes": packed_peak_reserved,
                "packed_serial_logprobs_match": True,
                "detached_export_matches": True,
                "publication_optimizer_step": receipt.optimizer_step,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            for model_id in ids:
                backend.unload_model(model_id)
        return {
            "benchmark_kind": "synchronized packed backend",
            "model": model_name,
            "torch": torch.__version__,
            "torchtitan_commit": TORCHTITAN_COMMIT,
            "gpu": torch.cuda.get_device_name(),
            "config": asdict(config),
            "results": results,
        }
    finally:
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--flavor", default="27B")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--adapter-counts", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--compile-layers", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.model,
        args.flavor,
        args.sequence_length,
        tuple(map(int, args.adapter_counts.split(","))),
        args.steps,
        args.compile_layers,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
