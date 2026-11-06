"""Actor diagnostics used only by the trainer memory-snapshot experiment."""

import hashlib
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from spindle.backends.miles_runtime.actor import SpindleMilesTrainRayActor


def digest(value):
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                result.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.update(pickle.dumps(item))

    visit(value)
    return result.hexdigest()


class SnapshotProbeActor(SpindleMilesTrainRayActor):
    def snapshot_suspend_comms(self):
        from torch.distributed.distributed_c10d import _world

        # The driver submits no further trainer commands until resume. All
        # earlier commands have completed before this actor RPC is dispatched.
        assert not hasattr(self, "_snapshot_suspended_comms")
        device = torch.device("cuda", torch.cuda.current_device())
        groups = [
            (name, group)
            for group, name in sorted(_world.pg_names.items(), key=lambda pair: pair[1])
            if "nccl" in str(dist.get_backend(group))
        ]
        assert groups, "No NCCL groups found"
        for _, group in groups:
            dist.barrier(group=group)
        torch.cuda.synchronize()
        self._snapshot_suspended_comms = []
        started = time.monotonic()
        for name, group in groups:
            backend = group._get_backend(device)
            print(
                f"SNAPSHOT_COMM suspending rank={dist.get_rank()} group={name}",
                flush=True,
            )
            backend.suspend()
            self._snapshot_suspended_comms.append((name, backend))
        torch.cuda.synchronize()
        return {
            "rank": dist.get_rank(),
            "groups": [name for name, _ in self._snapshot_suspended_comms],
            "elapsed_s": time.monotonic() - started,
        }

    def snapshot_resume_comms(self):
        started = time.monotonic()
        groups = self._snapshot_suspended_comms
        for name, backend in groups:
            print(
                f"SNAPSHOT_COMM resuming rank={dist.get_rank()} group={name}",
                flush=True,
            )
            backend.resume()
        torch.cuda.synchronize()
        del self._snapshot_suspended_comms
        return {
            "rank": dist.get_rank(),
            "groups": [name for name, _ in groups],
            "elapsed_s": time.monotonic() - started,
        }

    def snapshot_probe(self):
        # Exercise every existing process group, including lazily initialized
        # groups, before snapshotting and again after restoring.
        groups = []
        from torch.distributed.distributed_c10d import _world

        for group, name in sorted(_world.pg_names.items(), key=lambda pair: pair[1]):
            ranks = dist.get_process_group_ranks(group)
            backend = str(dist.get_backend(group))
            device = torch.cuda.current_device() if "nccl" in backend else "cpu"
            value = torch.tensor([dist.get_rank() + 1.0], device=device)
            dist.all_reduce(value, group=group)
            actual = value.item()
            expected = sum(rank + 1 for rank in ranks)
            assert actual == expected, (name, ranks, actual, expected)
            groups.append({"name": name, "backend": backend, "ranks": ranks})
        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]
        state = [
            {
                "parameters": dict(model.named_parameters()),
                "buffers": dict(model.named_buffers()),
                "gradients": {
                    name: (parameter.grad, getattr(parameter, "main_grad", None))
                    for name, parameter in model.named_parameters()
                },
            }
            for model in models
        ]
        optimizers = {
            slot: optimizer._inner.state_dict()
            for slot, optimizer in self.slot_optimizers.items()
        }
        torch.cuda.synchronize()
        return {
            "rank": dist.get_rank(),
            "pid": os.getpid(),
            "groups": groups,
            "model_sha256": digest(state),
            "optimizer_sha256": digest(optimizers),
            "rng_sha256": digest(
                (
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(),
                    get_cuda_rng_tracker().get_states(),
                    random.getstate(),
                    np.random.get_state(),
                )
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "allocated_bytes": torch.cuda.memory_allocated(),
        }
