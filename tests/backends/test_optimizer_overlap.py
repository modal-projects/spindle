import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from spindle.backends.miles_runtime.optimizer_overlap import (
    REMOTE_FAILURE,
    STREAM_POOL_ENV,
    detached_slot_state,
    optim_step_group,
)


@pytest.mark.parametrize("failure", [False, True])
def test_detached_gradients_survive_shared_buffer_reset_and_restore_views(failure):
    buffer = torch.tensor([2.0, 3.0, 5.0, 7.0])
    active = torch.nn.Parameter(torch.ones(2))
    other = torch.nn.Parameter(torch.ones(2))
    active.main_grad, other.main_grad = buffer[:2], buffer[2:]
    active.grad = torch.tensor([11.0, 13.0])
    original_main, original_grad = active.main_grad, active.grad
    copies = []
    with (
        pytest.raises(RuntimeError, match="optimizer failed")
        if failure
        else nullcontext()
    ):
        with detached_slot_state({0: [active]}) as copies:
            assert torch.equal(active.main_grad, torch.tensor([2.0, 3.0]))
            assert torch.equal(active.grad, torch.tensor([11.0, 13.0]))
            assert torch.equal(other.main_grad, torch.tensor([5.0, 7.0]))
            buffer.zero_()
            other.main_grad.add_(17)
            assert torch.equal(active.main_grad, torch.tensor([2.0, 3.0]))
            if failure:
                raise RuntimeError("optimizer failed")
    assert active.main_grad is original_main
    assert active.grad is original_grad
    assert torch.equal(other.main_grad, torch.tensor([17.0, 17.0]))
    assert torch.equal(copies.gradients[0], torch.tensor([2.0, 3.0]))
    assert torch.equal(copies.gradients[1], torch.tensor([11.0, 13.0]))


def test_detachment_restores_parameters_when_cloning_fails():
    parameter = torch.nn.Parameter(torch.ones(2))
    main = parameter.main_grad = torch.tensor([3.0, 4.0])
    broken = SimpleNamespace(main_grad=object(), grad=None, data=torch.ones(1))
    with pytest.raises(AttributeError):
        with detached_slot_state({0: [parameter, broken]}):
            pass
    assert parameter.main_grad is main


class FakeCuda:
    """Side streams whose queued kernels run when the stream drains."""

    def __init__(self, monkeypatch, pool_size, parameters_by_slot):
        self.trace = []
        self.streams = []
        self.current = None
        monkeypatch.setenv(STREAM_POOL_ENV, str(pool_size))
        monkeypatch.setattr(torch.cuda, "Stream", self._new_stream)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: "compute")
        monkeypatch.setattr(torch.cuda, "stream", self._use)
        module = ModuleType("miles.backends.megatron_utils.lora.optimizer")
        module.adapter_slot_parameters = lambda model, slot: parameters_by_slot[slot]
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def _new_stream(self):
        harness = self

        class Stream:
            index = len(self.streams)
            queued = []

            def wait_stream(self, compute):
                harness.trace.append(("wait", self.index))

            def synchronize(self):
                harness.trace.append(("drain", self.index))
                while self.queued:
                    self.queued.pop(0)()

        stream = Stream()
        self.streams.append(stream)
        return stream

    @contextmanager
    def _use(self, stream):
        self.current = stream
        try:
            yield
        finally:
            self.current = None

    def enqueue(self, action):
        self.current.queued.append(action)


class QueuedOptimizer:
    """Enqueues its update on the current stream like an async CUDA kernel."""

    def __init__(self, cuda, slot, parameter, *, fail=False):
        self.cuda, self.slot, self.parameter, self.fail = cuda, slot, parameter, fail

    def apply_adam_params(self, params):
        self.lr = params["learning_rate"]

    def prepare_grads(self):
        self.gradient = self.parameter.main_grad

    def clip_and_step(self, norm):
        self.cuda.trace.append(("norm", self.slot))
        if self.fail:
            raise RuntimeError("boom")
        weight, lr, gradient = self.parameter.data, self.lr, self.gradient
        self.cuda.enqueue(lambda: weight.sub_(lr * gradient))
        return {"grad_norm": 2.0}

    def zero_grads(self):
        gradient = self.parameter.main_grad
        self.cuda.enqueue(lambda: gradient.zero_())

    def allgather_params(self):
        assert torch.equal(self.parameter, torch.tensor([0.8]))
        self.cuda.trace.append(("gather", self.slot))


def _group(cuda, parameters, *, fail=(), forward_fails=False):
    shared_buffers = [parameter.main_grad for parameter in parameters.values()]

    def forward(**kwargs):
        cuda.trace.append("forward")
        for parameter, shared in zip(parameters.values(), shared_buffers, strict=True):
            assert torch.equal(parameter, torch.tensor([1.0]))
            shared.fill_(7.0)
        if forward_fails:
            raise RuntimeError("forward failed")
        return {"per_datum": []}

    actor = SimpleNamespace(
        model=[],
        slot_optimizers={
            slot: QueuedOptimizer(cuda, slot, parameter, fail=slot in fail)
            for slot, parameter in parameters.items()
        },
        forward_backward=forward,
    )
    adam = {slot: {"learning_rate": 0.1, "grad_clip_norm": 1.0} for slot in parameters}
    return optim_step_group(
        actor, adam_params_by_slot=adam, batch_id=1, rollout_data_ref="rows"
    )


def _parameters(slots):
    parameters = {}
    for slot in slots:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.main_grad = torch.tensor([2.0])
        parameters[slot] = [parameter]
    return parameters


@pytest.mark.parametrize("forward_fails", [False, True])
def test_group_shares_a_stream_pool_and_commits_after_the_forward(
    monkeypatch, forward_fails
):
    parameters = _parameters((3, 1, 2))
    cuda = FakeCuda(monkeypatch, 2, parameters)
    with (
        pytest.raises(RuntimeError, match="forward failed")
        if forward_fails
        else nullcontext()
    ):
        result = _group(
            cuda,
            {slot: parameters[slot][0] for slot in parameters},
            forward_fails=forward_fails,
        )
        assert result["optimizer_outcomes"] == {
            slot: {"grad_norm": 2.0} for slot in (1, 2, 3)
        }
    assert len(cuda.streams) == 2
    assert cuda.trace == [
        ("wait", 0),
        ("norm", 1),
        ("wait", 1),
        ("norm", 2),
        ("wait", 0),
        ("norm", 3),
        "forward",
        ("drain", 0),
        ("drain", 1),
    ] + ([] if forward_fails else [("gather", 1), ("gather", 2), ("gather", 3)])
    expected = 1.0 if forward_fails else 0.8
    for parameter in parameters.values():
        assert torch.equal(parameter[0], torch.tensor([expected]))
        assert torch.equal(parameter[0].main_grad, torch.tensor([7.0]))


@pytest.mark.parametrize("pool_size", [1, 4])
@pytest.mark.parametrize("remote", [False, True])
def test_failed_slot_is_isolated_and_collectives_keep_slot_order(
    monkeypatch, pool_size, remote
):
    parameters = _parameters((0, 1, 2))
    cuda = FakeCuda(monkeypatch, pool_size, parameters)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(flag, op):
        if remote and len([item for item in cuda.trace if item[0] == "agree"]) == 1:
            flag.fill_(1)
        cuda.trace.append(("agree", int(flag.item())))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    result = _group(
        cuda,
        {slot: parameters[slot][0] for slot in parameters},
        fail=() if remote else (1,),
    )
    assert result["optimizer_outcomes"] == {
        0: {"grad_norm": 2.0},
        1: {"error": REMOTE_FAILURE if remote else "RuntimeError: boom"},
        2: {"grad_norm": 2.0},
    }
    collectives = [
        item for item in cuda.trace if item[0] in ("norm", "agree", "gather")
    ]
    assert collectives == [
        ("norm", 0),
        ("agree", 0),
        ("norm", 1),
        ("agree", 1),
        ("norm", 2),
        ("agree", 0),
        ("gather", 0),
        ("gather", 2),
    ]
    assert torch.equal(parameters[0][0], torch.tensor([0.8]))
    assert torch.equal(parameters[1][0], torch.tensor([1.0]))
    assert torch.equal(parameters[2][0], torch.tensor([0.8]))


class BatchedOptimizer(QueuedOptimizer):
    @classmethod
    def step_many(cls, requests):
        outcomes = []
        for optimizer, params in requests:
            optimizer.cuda.trace.append(("step_many", optimizer.slot))
            optimizer.apply_adam_params(params)
            optimizer.prepare_grads()
            outcomes.append(optimizer.clip_and_step(params["grad_clip_norm"]))
            optimizer.zero_grads()
        return outcomes


@pytest.mark.parametrize("mixed", [False, True])
def test_shared_step_many_runs_once_per_group(monkeypatch, mixed):
    parameters = _parameters([2, 0, 1])
    cuda = FakeCuda(monkeypatch, 4, parameters)
    classes = {0: BatchedOptimizer, 1: BatchedOptimizer, 2: BatchedOptimizer}
    if mixed:
        classes[1] = QueuedOptimizer
    actor = SimpleNamespace(
        model=[],
        slot_optimizers={
            slot: classes[slot](cuda, slot, parameter[0])
            for slot, parameter in parameters.items()
        },
    )
    adam = {slot: {"learning_rate": 0.1, "grad_clip_norm": 1.0} for slot in parameters}
    result = optim_step_group(actor, adam_params_by_slot=adam)
    assert result["optimizer_outcomes"] == {
        slot: {"grad_norm": 2.0} for slot in range(3)
    }
    batched = [slot for kind, slot in cuda.trace if kind == "step_many"]
    assert batched == ([] if mixed else [0, 1, 2])
    assert [slot for kind, slot in cuda.trace if kind == "norm"] == [0, 1, 2]
    assert [item for item in cuda.trace if item[0] == "gather"] == [
        ("gather", slot) for slot in range(3)
    ]
    for parameter in parameters.values():
        assert torch.equal(parameter[0], torch.tensor([0.8]))


def test_failing_step_many_fails_every_slot_and_restores_staging(monkeypatch):
    parameters = _parameters([0, 1])
    cuda = FakeCuda(monkeypatch, 2, parameters)
    actor = SimpleNamespace(
        model=[],
        slot_optimizers={
            slot: BatchedOptimizer(cuda, slot, parameter[0], fail=slot == 1)
            for slot, parameter in parameters.items()
        },
    )
    adam = {slot: {"learning_rate": 0.1, "grad_clip_norm": 1.0} for slot in parameters}
    result = optim_step_group(actor, adam_params_by_slot=adam)
    assert result["optimizer_outcomes"] == {
        slot: {"error": "RuntimeError: boom"} for slot in range(2)
    }
    assert not [item for item in cuda.trace if item[0] == "gather"]
    for parameter in parameters.values():
        assert torch.equal(parameter[0], torch.tensor([1.0]))


class MomentumOptimizer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.momentum = [torch.zeros_like(parameter) for parameter in parameters]

    def apply_adam_params(self, params):
        self.lr = params["learning_rate"]

    def prepare_grads(self):
        self.grads = [parameter.main_grad for parameter in self.parameters]

    def clip_and_step(self, clip):
        norm = torch.sqrt(sum((grad * grad).sum() for grad in self.grads))
        scale = min(1.0, clip / (float(norm) + 1e-6))
        for parameter, grad, momentum in zip(
            self.parameters, self.grads, self.momentum, strict=True
        ):
            momentum.mul_(0.9).add_(grad * scale)
            parameter.data.sub_(self.lr * momentum)
        return {"grad_norm": float(norm)}

    def zero_grads(self):
        for grad in self.grads:
            grad.zero_()

    def allgather_params(self):
        pass


def test_group_matches_sequential_steps_bitwise(monkeypatch):
    generator = torch.Generator().manual_seed(0)
    parameters = {}
    for slot in range(3):
        slot_parameters = []
        for shape in ((4, 3), (3,)):
            parameter = torch.nn.Parameter(torch.randn(*shape, generator=generator))
            parameter.main_grad = torch.randn(*shape, generator=generator)
            slot_parameters.append(parameter)
        parameters[slot] = slot_parameters
    adam = {
        slot: {"learning_rate": 0.01 * (slot + 1), "grad_clip_norm": 0.5 + slot}
        for slot in parameters
    }
    sequential = {}
    for slot, slot_parameters in parameters.items():
        sequential[slot] = []
        for parameter in slot_parameters:
            clone = torch.nn.Parameter(parameter.detach().clone())
            clone.main_grad = parameter.main_grad.clone()
            sequential[slot].append(clone)
    sequential_optimizers = {
        slot: MomentumOptimizer(sequential[slot]) for slot in sequential
    }
    for slot in sorted(sequential):
        optimizer = sequential_optimizers[slot]
        optimizer.apply_adam_params(adam[slot])
        optimizer.prepare_grads()
        optimizer.clip_and_step(adam[slot]["grad_clip_norm"])
        optimizer.zero_grads()

    FakeCuda(monkeypatch, 2, parameters)
    optimizers = {slot: MomentumOptimizer(parameters[slot]) for slot in parameters}
    actor = SimpleNamespace(model=[], slot_optimizers=optimizers)
    result = optim_step_group(actor, adam_params_by_slot=adam)
    assert result.keys() == {"optimizer_outcomes"}
    for slot in parameters:
        for grouped, expected in zip(parameters[slot], sequential[slot], strict=True):
            assert torch.equal(grouped, expected)
            assert torch.equal(grouped.main_grad, expected.main_grad)
        for grouped, expected in zip(
            optimizers[slot].momentum, sequential_optimizers[slot].momentum, strict=True
        ):
            assert torch.equal(grouped, expected)
