import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


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
    sys.modules["megatron.core"].parallel_state = module("megatron.core.parallel_state")
    module(
        "megatron.core.dist_checkpointing.mapping",
        ShardedTensorFactory=ShardedTensorFactory,
    )
    lora = module("miles.backends.megatron_utils.lora")
    lora.checkpoint = module("miles.backends.megatron_utils.lora.checkpoint")
    module(
        "miles.backends.megatron_utils.lora.slots",
        slice_lora_to_rank=lambda name, tensor, rank: tensor,
    )
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


def _gathered():
    return {
        (0, "gate_up_proj", "lora_A"): _experts(0, (32, 4, 6)),
        (0, "gate_up_proj", "lora_B"): _experts(1, (32, 12, 4)),
        (0, "down_proj", "lora_A"): _experts(2, (32, 4, 6)),
        (0, "down_proj", "lora_B"): _experts(3, (32, 6, 4)),
    }


def _name(projection, side):
    return f"base_model.model.model.layers.0.mlp.experts.{projection}.{side}.weight"


def test_rank_zero_shard_is_replaced_by_every_expert(compat) -> None:
    gathered = _gathered()
    published = {
        _name(projection, side): tensor[:4].unsqueeze(0)
        for (_, projection, side), tensor in gathered.items()
    }
    attention = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    published[attention] = torch.ones(4, 6)

    fixed, replaced = compat.full_expert_tensors(published, gathered)

    assert replaced == 4
    for (_, projection, side), tensor in gathered.items():
        assert torch.equal(fixed[_name(projection, side)], tensor)
    assert fixed[attention] is published[attention]


def test_complete_export_is_left_alone(compat) -> None:
    gathered = _gathered()
    published = {
        _name(projection, side): tensor.clone()
        for (_, projection, side), tensor in gathered.items()
    }

    fixed, replaced = compat.full_expert_tensors(published, gathered)

    assert replaced == 0
    assert fixed == published


def test_export_that_disagrees_with_the_trainer_is_rejected(compat) -> None:
    gathered = _gathered()
    published = {
        _name(projection, side): tensor[4:8].unsqueeze(0)
        for (_, projection, side), tensor in gathered.items()
    }

    with pytest.raises(ValueError, match="do not match"):
        compat.full_expert_tensors(published, gathered)


def test_export_missing_an_expert_tensor_is_rejected(compat) -> None:
    gathered = _gathered()
    published = {
        _name("gate_up_proj", "lora_A"): gathered[(0, "gate_up_proj", "lora_A")]
    }

    with pytest.raises(ValueError, match="missing"):
        compat.full_expert_tensors(published, gathered)


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
