import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spindle.backends.miles_runtime.lora_init import install_expert_lora_a_init

EXPERTS, RANK, IN_FEATURES = 4, 32, 2880


class MultiLoRALinear(nn.Module):
    def __init__(self, init_method: str, shape, input_is_parallel: bool = False):
        super().__init__()
        self._column_init_method = init_method
        self.input_is_parallel = input_is_parallel
        self.adapters = nn.ModuleList(nn.Module() for _ in range(2))
        for adapter in self.adapters:
            adapter.linear_in = nn.Module()
            adapter.linear_in.weight = nn.Parameter(torch.empty(shape))
            self._draw(adapter.linear_in.weight)

    @staticmethod
    def _draw(weight):
        nn.init.kaiming_uniform_(weight.data, a=math.sqrt(5))

    def reset_adapter(self, idx: int) -> None:
        self._draw(self.adapters[idx].linear_in.weight)


class MultiLoRAGroupedExpertLinear(MultiLoRALinear):
    def reset_adapter(self, idx: int) -> None:
        self._draw(self.adapters[idx].linear_in.weight)


def _install(etp: int = 1):
    class MultiLoRA:
        def __call__(self, model, training: bool = True):
            return model

    layers = SimpleNamespace(
        MultiLoRAGroupedExpertLinear=MultiLoRAGroupedExpertLinear,
        _iter_multi_lora_modules=lambda model: (
            m for m in model.modules() if isinstance(m, MultiLoRALinear)
        ),
    )
    original_reset = MultiLoRAGroupedExpertLinear.reset_adapter
    install_expert_lora_a_init(
        multi_lora=SimpleNamespace(MultiLoRA=MultiLoRA),
        multi_lora_layers=layers,
        parallel_state=SimpleNamespace(
            get_expert_tensor_parallel_world_size=lambda: etp
        ),
    )
    return MultiLoRA(), original_reset


@pytest.fixture(autouse=True)
def _restore_reset():
    reset = MultiLoRAGroupedExpertLinear.reset_adapter
    yield
    MultiLoRAGroupedExpertLinear.reset_adapter = reset


def _bound_ratio(weight, in_features):
    return weight.abs().max().item() * math.sqrt(in_features)


def test_expert_lora_a_matches_per_expert_peft_bound_after_model_build():
    torch.manual_seed(0)
    transform, _ = _install()
    expert = MultiLoRAGroupedExpertLinear("kaiming", (EXPERTS, RANK, IN_FEATURES))
    dense = MultiLoRALinear("kaiming", (RANK, IN_FEATURES))
    dense_before = dense.adapters[0].linear_in.weight.detach().clone()
    model = nn.ModuleList([expert, dense])

    assert _bound_ratio(expert.adapters[0].linear_in.weight, IN_FEATURES) < 0.2
    transform(model)

    for adapter in expert.adapters:
        weight = adapter.linear_in.weight
        assert _bound_ratio(weight, IN_FEATURES) == pytest.approx(1.0, abs=1e-3)
        assert weight.std().item() == pytest.approx(
            1 / math.sqrt(3 * IN_FEATURES), rel=0.01
        )
    torch.testing.assert_close(dense.adapters[0].linear_in.weight, dense_before)


def test_expert_slot_reset_keeps_per_expert_peft_bound():
    torch.manual_seed(0)
    _install()
    expert = MultiLoRAGroupedExpertLinear("kaiming", (EXPERTS, RANK, IN_FEATURES))
    expert.reset_adapter(1)
    weight = expert.adapters[1].linear_in.weight
    assert _bound_ratio(weight, IN_FEATURES) == pytest.approx(1.0, abs=1e-3)


def test_row_parallel_expert_uses_full_input_width():
    torch.manual_seed(0)
    transform, _ = _install(etp=2)
    local_in = IN_FEATURES // 2
    expert = MultiLoRAGroupedExpertLinear(
        "kaiming", (EXPERTS, RANK, local_in), input_is_parallel=True
    )
    transform(nn.ModuleList([expert]))
    weight = expert.adapters[0].linear_in.weight
    assert _bound_ratio(weight, IN_FEATURES) == pytest.approx(1.0, abs=1e-3)


def test_non_kaiming_expert_init_is_untouched():
    torch.manual_seed(0)
    transform, _ = _install()
    expert = MultiLoRAGroupedExpertLinear("xavier", (EXPERTS, RANK, IN_FEATURES))
    before = expert.adapters[0].linear_in.weight.detach().clone()
    transform(nn.ModuleList([expert]))
    expert.reset_adapter(1)
    torch.testing.assert_close(expert.adapters[0].linear_in.weight, before)


def test_install_is_idempotent():
    torch.manual_seed(0)
    _, original = _install()
    wrapped = MultiLoRAGroupedExpertLinear.reset_adapter
    _install()
    assert MultiLoRAGroupedExpertLinear.reset_adapter is wrapped
    assert wrapped.__wrapped__ is original
