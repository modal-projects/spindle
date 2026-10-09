import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def compat(monkeypatch):
    def module(name, **attributes):
        value = types.ModuleType(name)
        for attribute, member in attributes.items():
            setattr(value, attribute, member)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    class ShardedTensorFactory:
        def __init__(self, data):
            self.data = data
            self.key = "unused"

    module("megatron", core=module("megatron.core"))
    sys.modules["megatron.core"].dist_checkpointing = module(
        "megatron.core.dist_checkpointing"
    )
    module(
        "megatron.core.dist_checkpointing.mapping",
        ShardedTensorFactory=ShardedTensorFactory,
    )
    lora = module("miles.backends.megatron_utils.lora")
    lora.checkpoint = module("miles.backends.megatron_utils.lora.checkpoint")
    path = (
        Path(__file__).parents[2]
        / "src/spindle/backends/miles_runtime/expert_lora_compat.py"
    )
    spec = importlib.util.spec_from_file_location("_test_expert_lora_compat", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.ShardedTensorFactory = ShardedTensorFactory
    return loaded


def _experts(seed: int, shape: tuple[int, ...]):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(torch.bfloat16)


def _restore(compat, monkeypatch, *, param, loaded, load_optimizer):
    events = []
    factory = compat.ShardedTensorFactory(param)
    plain = types.SimpleNamespace(data=torch.zeros(2), key="plain")
    checkpoint = compat.checkpoint
    checkpoint._WEIGHTS_KEY = "weights"
    checkpoint._slot_weights_sharded_state_dict = lambda model, slot: {
        "fc1": factory,
        "qkv": plain,
    }
    checkpoint._canonicalize_slot_keys = lambda tree, slot: tree

    def load(shells, path):
        events.append(("load", sorted(shells["weights"]), path))
        return {"weights": {"fc1": loaded}}

    monkeypatch.setattr(compat.dist_checkpointing, "load", load, raising=False)
    monkeypatch.setattr(compat.dist, "get_rank", lambda: 0)
    optimizer = types.SimpleNamespace(
        slot=3, reload_masters=lambda: events.append("reload_masters")
    )
    compat.restore_factory_weights(
        [], optimizer, "/ckpt/state", load_optimizer=load_optimizer
    )
    return events


def test_factory_tensors_are_copied_into_the_params(compat, monkeypatch) -> None:
    param = torch.zeros(4, 12, 4, dtype=torch.bfloat16)
    loaded = _experts(5, (4, 12, 4))

    events = _restore(
        compat, monkeypatch, param=param, loaded=loaded, load_optimizer=True
    )

    assert torch.equal(param, loaded)
    assert events == [("load", ["fc1"], "/ckpt/state")]


def test_weights_only_load_refreshes_masters_after_the_copy(
    compat, monkeypatch
) -> None:
    param = torch.zeros(4, 12, 4, dtype=torch.bfloat16)
    loaded = _experts(6, (4, 12, 4))

    events = _restore(
        compat, monkeypatch, param=param, loaded=loaded, load_optimizer=False
    )

    assert torch.equal(param, loaded)
    assert events[-1] == "reload_masters"


def test_loaded_shape_mismatch_is_rejected(compat) -> None:
    with pytest.raises(ValueError, match="shape"):
        compat.copy_loaded_into_params(
            {"fc1": torch.zeros(4, 12, 4)}, {"fc1": torch.zeros(4, 6, 4)}
        )
