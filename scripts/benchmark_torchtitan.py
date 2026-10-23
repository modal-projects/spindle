"""Packed-backend correctness and throughput; not asynchronous SDK latency."""

import argparse
from dataclasses import asdict
import json
import math
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


def check_packed_gradients(backend, items, *, validate=True):
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
        if validate:
            assert relative_error < 0.025, (
                f"packed gradients differ for {item.model_id}: {relative_error}"
            )
        errors[item.model_id] = relative_error
        state.optimizer.zero_grad(set_to_none=True)
        state.accumulating = False
    return errors


def logprob_difference(packed, serial):
    packed = torch.tensor(packed.loss_fn_outputs[0]["logprobs"].data)
    serial = torch.tensor(serial.loss_fn_outputs[0]["logprobs"].data)
    return tensor_difference(packed, serial)


def tensor_difference(packed, serial):
    packed, serial = packed.float(), serial.float()
    difference = packed - serial
    return {
        "max_absolute": difference.abs().max().item(),
        "mean_absolute": difference.abs().mean().item(),
        "relative_l2": (difference.norm() / serial.norm()).item(),
    }


def check_packed_logprobs(backend, items, sequence_length):
    def forward(rows):
        return backend.forward_backward(
            ForwardBatch(rows, "cross_entropy", forward_only=True)
        )

    packed = forward(items)
    adapter_errors = [
        logprob_difference(output, forward((item,))[0])
        for item, output in zip(items, packed, strict=True)
    ]
    saved_scales = [dict(module.scales) for module in backend.projections.values()]
    try:
        for module in backend.projections.values():
            module.scales = dict.fromkeys(module.scales, 0.0)
        base_packed = forward(items)
        base_errors = [
            logprob_difference(output, forward((item,))[0])
            for item, output in zip(items, base_packed, strict=True)
        ]
    finally:
        for module, scales in zip(
            backend.projections.values(), saved_scales, strict=True
        ):
            module.scales = scales
    results = {}
    for index, (item, adapter, base) in enumerate(
        zip(items, adapter_errors, base_errors, strict=True)
    ):
        # Native BF16 Qwen3.8-27B itself changes slightly with batch size.
        # Bound that baseline, then require LoRA not to materially worsen it.
        assert base["relative_l2"] < 0.005 and base["max_absolute"] < 1.0, base
        assert adapter["relative_l2"] <= 1.25 * base["relative_l2"] + 1e-4, (
            item.model_id,
            adapter,
            base,
        )
        assert adapter["max_absolute"] <= 1.25 * base["max_absolute"] + 0.05, (
            item.model_id,
            adapter,
            base,
        )
        # At a fixed batch shape, changing other clients must have NO effect.
        changed = tuple(
            row
            if i == index
            else ForwardItem(row.model_id, (make_data(100 + i, sequence_length),))
            for i, row in enumerate(items)
        )
        token_isolation = logprob_difference(packed[index], forward(changed)[index])
        assert token_isolation["max_absolute"] == 0, token_isolation
        try:
            for module in backend.projections.values():
                module.scales = {
                    slot: scale if slot == backend.jobs[item.model_id].slot else 0.0
                    for slot, scale in module.scales.items()
                }
            adapter_isolation = logprob_difference(packed[index], forward(items)[index])
            assert adapter_isolation["max_absolute"] == 0, adapter_isolation
        finally:
            for module, scales in zip(
                backend.projections.values(), saved_scales, strict=True
            ):
                module.scales = scales
        results[item.model_id] = {
            "adapter_error": adapter,
            "frozen_base_error": base,
            "companion_tokens_max_error": token_isolation["max_absolute"],
            "companion_adapters_max_error": adapter_isolation["max_absolute"],
        }
    return results


def diagnose_batching(backend, items, sequence_length):
    layer_outputs = []

    def capture(module, args, output):
        layer_outputs.append(output[:sequence_length].detach().cpu())

    handles = [
        layer.register_forward_hook(capture) for layer in backend.model.layers.values()
    ]

    def forward(rows):
        layer_outputs.clear()
        return backend.forward_backward(
            ForwardBatch(rows, "cross_entropy", forward_only=True)
        )

    result = {}
    for label in ("trained_adapters", "frozen_base"):
        if label == "frozen_base":
            saved_scales = [
                dict(module.scales) for module in backend.projections.values()
            ]
            for module in backend.projections.values():
                module.scales = dict.fromkeys(module.scales, 0.0)
        packed = forward(items)
        packed_layers = list(layer_outputs)
        result[label] = []
        for index, (item, output) in enumerate(zip(items, packed, strict=True)):
            result[label].append(logprob_difference(output, forward((item,))[0]))
            if index == 0:
                result[label + "_layers"] = [
                    tensor_difference(left, right)
                    for left, right in zip(packed_layers, layer_outputs, strict=True)
                ]
        if label == "frozen_base":
            for module, scales in zip(
                backend.projections.values(), saved_scales, strict=True
            ):
                module.scales = scales
    for handle in handles:
        handle.remove()
    layer_outputs.clear()
    packed = forward(items)
    result["changed_companion_tokens"] = []
    result["changed_companion_adapters"] = []
    for index, item in enumerate(items):
        changed = tuple(
            row
            if i == index
            else ForwardItem(row.model_id, (make_data(100 + i, sequence_length),))
            for i, row in enumerate(items)
        )
        result["changed_companion_tokens"].append(
            logprob_difference(packed[index], forward(changed)[index])
        )
        saved_scales = [dict(module.scales) for module in backend.projections.values()]
        for module in backend.projections.values():
            module.scales = {
                slot: scale if slot == backend.jobs[item.model_id].slot else 0.0
                for slot, scale in module.scales.items()
            }
        result["changed_companion_adapters"].append(
            logprob_difference(packed[index], forward(items)[index])
        )
        for module, scales in zip(
            backend.projections.values(), saved_scales, strict=True
        ):
            module.scales = scales
    result["gradient_relative_errors"] = check_packed_gradients(
        backend, items, validate=False
    )
    return result


def measure_training(backend, items, adam, steps, *, serial):
    phase = "serial" if serial else "packed"
    batches = (
        [ForwardBatch((item,), "cross_entropy") for item in items]
        if serial
        else [ForwardBatch(items, "cross_entropy")]
    )
    history = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        losses, norms = {}, {}
        for batch in batches:
            outputs = backend.forward_backward(batch)
            ids = tuple(item.model_id for item in batch.items)
            outcomes = backend.optim_step(ids, adam)
            for model_id, output, outcome in zip(ids, outputs, outcomes, strict=True):
                assert outcome.metrics["update_successful:mean"] == 1, model_id
                losses[model_id] = output.metrics["loss:sum"]
                norms[model_id] = outcome.metrics["grad_norm:mean"]
                assert math.isfinite(losses[model_id]), (model_id, losses[model_id])
        torch.cuda.synchronize()
        record = {
            "step": step,
            "step_s": time.perf_counter() - step_started,
            "loss_by_adapter": losses,
            "gradient_norm_by_adapter": norms,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        history.append(record)
        if step == 1 or step % 10 == 0 or step == steps:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "step": step,
                        "total_steps": steps,
                        "step_s": record["step_s"],
                        "mean_loss": sum(losses.values()) / len(losses),
                        "max_gradient_norm": max(norms.values()),
                        "allocated_gib": record["allocated_bytes"] / 2**30,
                    }
                ),
                flush=True,
            )
    return {
        "elapsed_s": time.perf_counter() - started,
        "history": history,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run(
    model_name,
    flavor,
    sequence_length,
    adapter_counts,
    steps,
    compile_layers,
    diagnose=False,
    max_batch_tokens=0,
    max_batch_rows=4,
):
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
        max_batch_rows=max_batch_rows,
        max_batch_tokens=max_batch_tokens or sequence_length * max(adapter_counts),
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
            if diagnose:
                reduced = (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                )
                diagnostic = diagnose_batching(backend, items, sequence_length)
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    False
                )
                precise = diagnose_batching(backend, items, sequence_length)
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    reduced
                )
                return {"default_reduction": diagnostic, "fp32_reduction": precise}
            logprob_checks = check_packed_logprobs(backend, items, sequence_length)
            gradient_errors = check_packed_gradients(backend, items)
            # Compare both schedules from the same warmed parameters and Adam state.
            for model_id in ids:
                backend.capture_checkpoint(
                    model_id,
                    model_id,
                    destination=model_id,
                    include_optimizer=True,
                )
                backend.persist_checkpoint(model_id, model_id, overwrite=True)
            print(f"Measuring {count} adapters", flush=True)
            # A publication must survive a subsequent optimizer update unchanged.
            receipt = backend.capture_sampler_snapshot(ids[0], "validation-export", 1)
            saved = {
                key: value.clone()
                for key, value in backend.sampler_captures["validation-export"][
                    2
                ].items()
            }
            packed_run = measure_training(backend, items, adam, steps, serial=False)
            assert all(backend.jobs[model_id].step == steps + 2 for model_id in ids)
            print("Checking correctness after packed training", flush=True)
            final_logprob_checks = check_packed_logprobs(
                backend, items, sequence_length
            )
            final_gradient_errors = check_packed_gradients(backend, items)
            for model_id in ids:
                backend.load_checkpoint(model_id, model_id, restore_optimizer=True)
                assert backend.jobs[model_id].step == 2
            serial_run = measure_training(backend, items, adam, steps, serial=True)
            assert all(backend.jobs[model_id].step == steps + 2 for model_id in ids)
            elapsed = packed_run["elapsed_s"]
            serial_elapsed = serial_run["elapsed_s"]
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
                "packed_serial_max_logprob_error": max(
                    check["adapter_error"]["max_absolute"]
                    for check in logprob_checks.values()
                ),
                "logprob_checks": logprob_checks,
                "peak_allocated_bytes": packed_run["peak_allocated_bytes"],
                "peak_reserved_bytes": packed_run["peak_reserved_bytes"],
                "packed_history": packed_run["history"],
                "serial_history": serial_run["history"],
                "serial_starts_from_same_checkpoint": True,
                "post_training_logprob_checks": final_logprob_checks,
                "post_training_gradient_relative_errors": final_gradient_errors,
                "packed_serial_logprobs_within_native_bf16_baseline": True,
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
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-batch-rows", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.model,
        args.flavor,
        args.sequence_length,
        tuple(map(int, args.adapter_counts.split(","))),
        args.steps,
        args.compile_layers,
        args.diagnose,
        args.max_batch_tokens,
        args.max_batch_rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
