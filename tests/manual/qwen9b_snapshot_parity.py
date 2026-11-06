"""Byte-level state audit and repeatable-step controls for the 9B snapshot probe."""

import copy
import gc
import hashlib
import random
import sys
import types
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from qwen9b_snapshot_probe import SnapshotProbeActor, digest


def cpu_bytes_hash(tensor):
    return hashlib.sha256(memoryview(tensor.numpy())).hexdigest()


def named_tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from named_tensors(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from named_tensors(child, f"{path}.{index}")


class SnapshotParityActor(SnapshotProbeActor):
    def parity_attention_backend(self):
        module = sys.modules.get(
            "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
        )
        if module is None:
            return {}
        cache = module._attention_backends
        return {
            key: str(cache.get(key))
            for key in (
                "use_flash_attention",
                "flash_attention_backend",
                "use_fused_attention",
                "fused_attention_backend",
            )
        }

    def parity_set_te_determinism(self, enabled):
        changed = []
        for model in self._parity_models():
            for name, module in model.named_modules():
                if "attention" in type(module).__name__.lower() and isinstance(
                    getattr(module, "deterministic", None), bool
                ):
                    module.deterministic = enabled
                    changed.append(name)
        assert changed, "No attention determinism switches found"
        module = sys.modules[
            "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
        ]
        module._attention_backends["backend_selection_requires_update"] = True
        return {"rank": dist.get_rank(), "enabled": enabled, "modules": changed}

    def _parity_models(self):
        return self.model if isinstance(self.model, (list, tuple)) else [self.model]

    def _parity_named(self):
        for index, model in enumerate(self._parity_models()):
            for name, param in model.named_parameters(remove_duplicate=False):
                yield f"model.{index}.parameter.{name}", param
                for attr in ("grad", "main_grad", "main_param"):
                    value = getattr(param, attr, None)
                    if isinstance(value, torch.Tensor):
                        yield f"model.{index}.{attr}.{name}", value
            for name, buf in model.named_buffers(remove_duplicate=False):
                yield f"model.{index}.buffer.{name}", buf
        for slot, optimizer in self.slot_optimizers.items():
            yield from named_tensors(
                optimizer._inner.state_dict(), f"optimizer.{slot}.state_dict"
            )
            for index, child in enumerate(optimizer._inner.chained_optimizers):
                for pindex, param in enumerate(child.get_parameters()):
                    yield f"optimizer.{slot}.child.{index}.master.{pindex}", param
                    if param.grad is not None:
                        yield (
                            f"optimizer.{slot}.child.{index}.master_grad.{pindex}",
                            param.grad,
                        )

    def parity_save(self):
        """Keep independent CPU byte copies, including full storage padding."""
        torch.cuda.synchronize()
        graph, seen, grads, discovered = [], set(), [], []

        def visit(value, path):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                discovered.append((path, value))
                if isinstance(value, torch.nn.Parameter):
                    grads.append((value, value.grad))
                if getattr(value, "__dict__", None):
                    visit(value.__dict__, path + ".attributes")
                return
            if isinstance(value, dict):
                graph.append((value, value.copy()))
                for index, (key, child) in enumerate(value.items()):
                    label = (
                        str(key)
                        if isinstance(key, (str, int, float))
                        else f"key_{index}"
                    )
                    visit(child, f"{path}.{label}")
            elif isinstance(value, list):
                graph.append((value, value.copy()))
                for index, child in enumerate(value):
                    visit(child, f"{path}.{index}")
            elif isinstance(value, tuple):
                for index, child in enumerate(value):
                    visit(child, f"{path}.{index}")
            elif not isinstance(
                value, (type, types.FunctionType, types.MethodType)
            ) and (
                isinstance(value, torch.optim.Optimizer)
                or type(value).__module__.startswith(
                    (
                        "torch.nn",
                        "torch.optim",
                        "megatron",
                        "miles",
                        "argparse",
                        "transformer_engine",
                        "apex",
                    )
                )
            ):
                if hasattr(value, "__dict__"):
                    visit(vars(value), path + ".attributes")

        visit(self.model, "model_graph")
        visit(self.slot_optimizers, "optimizer_graph")
        visit(self.args, "args")
        self._parity_graph = graph
        self._parity_param_grads = grads
        named = list(self._parity_named())
        # GC enumeration also finds live DDP flat buffers and library workspaces
        # which are not registered model parameters or buffers.
        physical = []
        for value in gc.get_objects():
            if type(value) in (torch.Tensor, torch.nn.Parameter) and value.is_cuda:
                physical.append(value)
        all_tensors = (
            named + discovered + [(f"gc_cuda.{i}", t) for i, t in enumerate(physical)]
        )
        storages, cpu_values, skipped = {}, {}, Counter()
        for name, tensor in all_tensors:
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                skipped[type(tensor).__name__] += 1
                continue
            if tensor.layout != torch.strided:
                raise RuntimeError(
                    f"Unsupported storage layout: {name}: {tensor.layout}"
                )
            if not tensor.is_cuda:
                if name.startswith(("model.", "optimizer.")):
                    cpu_values[name] = (tensor, tensor.detach().clone())
                continue
            storage = tensor.untyped_storage()
            size = storage.nbytes()
            key = (str(tensor.device), storage.data_ptr(), size)
            if key not in storages:
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                raw.set_(storage, 0, (size,), (1,))
                reference = raw.cpu()
                storages[key] = {
                    "raw": raw,
                    "reference": reference,
                    "labels": [],
                    "sha256": cpu_bytes_hash(reference),
                }
            if not name.startswith("gc_cuda."):
                storages[key]["labels"].append(name)
        self._parity_storages = storages
        self._parity_cpu_values = cpu_values
        self._parity_rng = copy.deepcopy(
            (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
                get_cuda_rng_tracker().get_states(),
                random.getstate(),
                np.random.get_state(),
            )
        )
        summary = {
            "rank": dist.get_rank(),
            "cuda_storages": len(storages),
            "cuda_storage_bytes": sum(key[2] for key in storages),
            "named_tensor_count": len(named),
            "cpu_named_tensors": len(cpu_values),
            "skipped_tensor_subclasses": dict(skipped),
            "attention_backend": self.parity_attention_backend(),
            "inventory": [
                {
                    "device": key[0],
                    "address": key[1],
                    "bytes": key[2],
                    "sha256": row["sha256"],
                    "labels": row["labels"],
                }
                for key, row in storages.items()
            ],
        }
        print(
            "PARITY_SAVED",
            {k: v for k, v in summary.items() if k != "inventory"},
            flush=True,
        )
        return summary

    def parity_audit(self):
        torch.cuda.synchronize()
        mismatches = []
        checked = 0
        for key, row in self._parity_storages.items():
            actual = row["raw"].cpu()
            checked += key[2]
            if not torch.equal(actual, row["reference"]):
                mismatches.append(
                    {
                        "address": key[1],
                        "bytes": key[2],
                        "labels": row["labels"],
                        "expected_sha256": row["sha256"],
                        "actual_sha256": cpu_bytes_hash(actual),
                    }
                )
        for name, (value, expected) in self._parity_cpu_values.items():
            if digest(value) != digest(expected):
                mismatches.append({"cpu_tensor": name})
        result = {
            "rank": dist.get_rank(),
            "byte_equal": not mismatches,
            "cuda_bytes_checked": checked,
            "mismatches": mismatches,
        }
        assert not mismatches, result
        return result

    def parity_reset(self):
        # Restore the same objects/storage in place: do not reconstruct the
        # optimizer, round masters from BF16, or change communicator references.
        for container, saved in self._parity_graph:
            if isinstance(container, dict):
                container.clear()
                container.update(saved)
            else:
                container[:] = saved
        for param, grad in self._parity_param_grads:
            param.grad = grad
        with torch.no_grad():
            for row in self._parity_storages.values():
                row["raw"].copy_(row["reference"])
            for value, expected in self._parity_cpu_values.values():
                value.copy_(expected)
        cpu, cuda, tracker, python, numpy = copy.deepcopy(self._parity_rng)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        get_cuda_rng_tracker().set_states(tracker)
        random.setstate(python)
        np.random.set_state(numpy)
        return self.parity_audit()

    def parity_observe(self, stage, label):
        values = {}
        if stage == "gradients":
            for index, model in enumerate(self._parity_models()):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        grad = getattr(param, "main_grad", None)
                        if grad is None:
                            grad = param.grad
                        if grad is not None:
                            values[f"model.{index}.{name}"] = (
                                grad.detach().float().cpu()
                            )
        else:
            for index, model in enumerate(self._parity_models()):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        values[f"model.{index}.{name}"] = param.detach().float().cpu()
            for slot, optimizer in self.slot_optimizers.items():
                for name, value in named_tensors(
                    optimizer._inner.state_dict(), f"optimizer.{slot}"
                ):
                    values[name] = value.detach().float().cpu()
        baselines = self.__dict__.setdefault("_parity_observations", {})
        reference_key = (stage, "fa3" if label.startswith("fa3_") else "primary")
        if reference_key not in baselines:
            baselines[reference_key] = {
                name: value.clone() for name, value in values.items()
            }
        reference = baselines[reference_key]
        details, different, max_error, sign_changes, elements = [], 0, 0.0, 0, 0
        categories = {}
        for name, value in values.items():
            expected = reference[name]
            delta = (value - expected).abs()
            count = int(torch.count_nonzero(delta))
            different += count
            elements += value.numel()
            error = delta.max().item() if value.numel() else 0.0
            max_error = max(max_error, error)
            changed_sign = int(torch.count_nonzero((value * expected < 0)))
            sign_changes += changed_sign
            category = "optimizer" if name.startswith("optimizer.") else "model"
            totals = categories.setdefault(
                category, {"elements": 0, "different": 0, "max_abs": 0.0}
            )
            totals["elements"] += value.numel()
            totals["different"] += count
            totals["max_abs"] = max(totals["max_abs"], error)
            if count:
                details.append(
                    {
                        "name": name,
                        "elements": value.numel(),
                        "different": count,
                        "max_abs": error,
                        "reference_max_abs": expected.abs().max().item(),
                        "sign_changes": changed_sign,
                    }
                )
        return {
            "rank": dist.get_rank(),
            "stage": stage,
            "label": label,
            "attention_backend": self.parity_attention_backend(),
            "sha256": digest(values),
            "elements": elements,
            "different_elements": different,
            "max_abs": max_error,
            "sign_changes": sign_changes,
            "different_tensors": len(details),
            "categories": categories,
            "top_differences": sorted(
                details, key=lambda d: d["max_abs"], reverse=True
            )[:20],
        }
