from dataclasses import replace
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchtitan")

import torch
from torch import nn
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData
from safetensors.torch import save_file
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8 import build_model_config
from torchtitan.models.common.linear import Linear
from spindle.backends.torchtitan_runtime.model import load_hf_weights

from spindle.backends.contract import ForwardBatch, ForwardItem, ModelSpec
from spindle.backends.torchtitan_config import TorchTitanConfig
from spindle.backends.torchtitan_lora import TorchTitanBackend
from spindle.backends.torchtitan_runtime.lora import (
    AdapterBatch,
    MultiLoRALinear,
    install_adapters,
)
from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig


class TinyDecoder(nn.Module):
    """Small causal-independent decoder for testing command semantics on CPU."""

    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(17, 8)
        layer = nn.Module()
        layer.feed_forward = nn.Module()
        layer.feed_forward.w1 = nn.Linear(8, 8, bias=False)
        self.layers = nn.ModuleDict({"0": layer})
        self.lm_head = nn.Linear(8, 17, bias=False)

    def _get_attention_metadata(self, positions):
        return None

    def forward(self, tokens, **kwargs):
        return self.layers["0"].feed_forward.w1(self.tok_embeddings(tokens))


def make_backend(tmp_path):
    torch.manual_seed(5)
    model = TinyDecoder()
    config = TorchTitanConfig(
        hf_checkpoint="",
        target_modules=("gate_proj",),
        max_lora_rank=4,
        max_batch_tokens=16,
    )
    batch, projections = install_adapters(
        model, config.target_modules, config.max_lora_rank
    )
    return TorchTitanBackend(
        config,
        base_model="test",
        model=model,
        adapter_batch=batch,
        projections=projections,
        checkpoint_dir=tmp_path,
    )


def spec(seed=1, rank=2):
    return ModelSpec(
        "test",
        "lora",
        LoraConfig(
            rank=rank, seed=seed, train_attn=False, train_mlp=True, train_unembed=False
        ),
    )


def datum(tokens=(1, 2, 3, 4), weight=1.0):
    return Datum(
        model_input=ModelInput.from_ints(list(tokens)),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=[*tokens[1:], 5], dtype="int64", shape=[len(tokens)]
            ),
            "weights": TensorData(
                data=[weight] * len(tokens), dtype="float32", shape=[len(tokens)]
            ),
        },
    )


def batch(*ids, forward_only=False):
    return ForwardBatch(
        tuple(
            ForwardItem(
                model_id,
                (
                    datum(),
                    datum((2, 3)),
                ),
            )
            for model_id in ids
        ),
        "cross_entropy",
        forward_only=forward_only,
    )


def test_batched_projection_matches_serial_outputs_and_gradients():
    torch.manual_seed(1)
    routing = AdapterBatch(("0", "1", "0"))
    layer = MultiLoRALinear(nn.Linear(5, 7), routing, 4)
    generator = torch.Generator().manual_seed(13)
    for key, rank in [("0", 2), ("1", 4), ("2", 3)]:
        layer.add_adapter(key, rank, 8, generator)
        nn.init.normal_(layer.b[key])
    inputs = torch.randn(3, 6, 5, requires_grad=True)
    actual = layer(inputs.flatten(0, 1)).view(3, 6, 7)
    reference = torch.stack(
        [
            layer.base(row)
            + (row @ layer.a[key].T @ layer.b[key].T) * layer.scales[key]
            for row, key in zip(inputs, routing.slots, strict=True)
        ]
    )
    torch.testing.assert_close(actual, reference)
    params = [inputs, layer.a["0"], layer.b["0"], layer.a["1"], layer.b["1"]]
    expected_grads = torch.autograd.grad(
        reference.square().sum(), params, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), params)
    for got, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, expected)
    assert layer.a["2"].grad is None and layer.b["2"].grad is None
    assert layer.base.weight.grad is None


def test_command_batch_matches_independent_clients(tmp_path):
    combined = make_backend(tmp_path / "combined")
    serial = make_backend(tmp_path / "serial")
    for backend in (combined, serial):
        backend.accept_model("a", spec())
        backend.accept_model("b", spec(2, 4))
    got = combined.forward_backward(batch("a", "b"))
    expected = (
        *serial.forward_backward(batch("a")),
        *serial.forward_backward(batch("b")),
    )
    for left, right in zip(got, expected, strict=True):
        assert left.metrics["loss:sum"] == pytest.approx(
            right.metrics["loss:sum"], rel=1e-6
        )
    for key in ("a", "b"):
        for p, q in zip(
            combined.parameters_for(combined.jobs[key].slot),
            serial.parameters_for(serial.jobs[key].slot),
            strict=True,
        ):
            torch.testing.assert_close(p.grad, q.grad)
    combined.optim_step(("a", "b"), AdamParams(learning_rate=0.01))
    serial.optim_step(("a",), AdamParams(learning_rate=0.01))
    serial.optim_step(("b",), AdamParams(learning_rate=0.01))
    for key in ("a", "b"):
        for p, q in zip(
            combined.parameters_for(combined.jobs[key].slot),
            serial.parameters_for(serial.jobs[key].slot),
            strict=True,
        ):
            torch.testing.assert_close(p, q)


def test_forward_only_accumulation_and_optimizer_isolation(tmp_path):
    backend = make_backend(tmp_path)
    backend.accept_model("a", spec())
    backend.accept_model("b", spec(2))
    backend.forward_backward(batch("a", forward_only=True))
    assert not backend.jobs["a"].accumulating
    assert all(p.grad is None for p in backend.parameters_for("0"))
    backend.forward_backward(batch("a", "b"))
    b_before = [p.detach().clone() for p in backend.parameters_for("1")]
    b_grads = [p.grad.clone() for p in backend.parameters_for("1")]
    a_grads = [p.grad.clone() for p in backend.parameters_for("0")]
    backend.forward_backward(batch("a"))
    for p, grad in zip(backend.parameters_for("0"), a_grads, strict=True):
        torch.testing.assert_close(p.grad, 2 * grad)
    backend.optim_step(("a",), AdamParams(learning_rate=0.03, weight_decay=0.1))
    for p, before, grad in zip(
        backend.parameters_for("1"), b_before, b_grads, strict=True
    ):
        torch.testing.assert_close(p, before)
        torch.testing.assert_close(p.grad, grad)
    assert backend.jobs["a"].step == 1 and backend.jobs["b"].step == 0
    assert all(
        p.grad is None
        for name, p in backend.model.named_parameters()
        if ".a." not in name and ".b." not in name
    )


def test_checkpoint_is_immutable_and_restores_optimizer_and_gradients(tmp_path):
    backend = make_backend(tmp_path)
    backend.accept_model("a", spec())
    adam = AdamParams(learning_rate=0.01)
    backend.forward_backward(batch("a"))
    backend.optim_step(("a",), adam)
    backend.forward_backward(batch("a"))
    backend.capture_checkpoint(
        "a", "saved", destination="step1", include_optimizer=True
    )
    before = [p.clone() for p in backend.parameters_for("0")]
    backend.optim_step(("a",), adam)
    after = [p.clone() for p in backend.parameters_for("0")]
    uri = backend.persist_checkpoint("saved", "step1")
    backend.load_checkpoint("a", uri, restore_optimizer=True)
    for p, expected in zip(backend.parameters_for("0"), before, strict=True):
        torch.testing.assert_close(p, expected)
    backend.optim_step(("a",), adam)
    for p, expected in zip(backend.parameters_for("0"), after, strict=True):
        torch.testing.assert_close(p, expected)
    assert backend.jobs["a"].step == 2


def test_nonfinite_gradients_skip_update(tmp_path):
    backend = make_backend(tmp_path)
    backend.accept_model("a", spec())
    backend.forward_backward(batch("a"))
    parameters = backend.parameters_for("0")
    before = [p.clone() for p in parameters]
    parameters[0].grad.fill_(float("nan"))
    (output,) = backend.optim_step(("a",), AdamParams())
    assert output.metrics["skipped_nonfinite:sum"] == 1
    for p, expected in zip(parameters, before, strict=True):
        torch.testing.assert_close(p, expected)
    assert backend.jobs["a"].step == 0


def test_adapter_export_is_exact_and_detached(tmp_path):
    backend = make_backend(tmp_path)
    backend.accept_model("a", spec())
    backend.forward_backward(batch("a"))
    backend.optim_step(("a",), AdamParams(learning_rate=0.01))
    receipt = backend.capture_sampler_snapshot("a", "v1", 1)
    _, _, exported, config = backend.sampler_captures["v1"]
    name, projection = next(iter(backend.projections.items()))
    x = torch.randn(3, 8)
    backend.adapter_batch.slots = ("0",)
    expected = (
        projection.base(x)
        + x
        @ exported[name + ".lora_A.weight"].T
        @ exported[name + ".lora_B.weight"].T
        * config["lora_alpha"]
        / config["r"]
    )
    torch.testing.assert_close(projection(x), expected)
    backend.adapter_batch.slots = ()
    backend.unload_model("a")
    backend.accept_model("b", spec(8))
    assert receipt.optimizer_step == 1
    assert not torch.equal(exported[name + ".lora_A.weight"], projection.a["0"][:2])


def test_validation_precedes_gradient_changes(tmp_path):
    backend = make_backend(tmp_path)
    backend.accept_model("a", spec())
    invalid = replace(batch("a"), loss_fn_config={"ignored_option": 1})
    with pytest.raises(ValueError, match="loss_fn_config"):
        backend.forward_backward(invalid)
    with pytest.raises(ValueError, match="accumulated gradients"):
        backend.optim_step(("a",), AdamParams())
    with pytest.raises(ValueError, match="inside checkpoint_dir"):
        backend._checkpoint_path("../escape")
    assert all(p.grad is None for p in backend.parameters_for("0"))


def test_recipe_serializes_and_routes_lora_settings():
    recipe = BaseConfig(
        name="titan-test",
        model="Qwen/Qwen3.8-27B",
        backend="torchtitan",
        trainer_max_clients_per_instance=4,
        torchtitan_cfg={"max_lora_rank": 16},
    )
    deployment = DeploymentConfig.create(recipe)
    restored = DeploymentConfig.model_validate_json(deployment.model_dump_json())
    assert restored.trainer_settings["torchtitan"]["max_lora_rank"] == 16
    assert restored.inference_settings["max_lora_rank"] == 16
    assert restored.inference_settings["enable_lora"]
    with pytest.raises(ValueError, match="one GPU"):
        DeploymentConfig.create(
            BaseConfig(**{**vars(recipe), "trainer_gpus_per_node": 2})
        )


def test_native_27b_builds_with_expected_adapter_targets():
    config = build_model_config("27B", parallelism=ParallelismConfig(), seq_len=16384)
    config.vision_encoder = None
    with torch.device("meta"):
        model = config.build()
        _, projections = install_adapters(
            model, TorchTitanConfig(hf_checkpoint="").target_modules, 32
        )
    assert len(model.layers) == 64
    assert sum(module.num_linears for module in projections.values()) == 256
    assert 26_000_000_000 < sum(p.numel() for p in model.parameters()) < 28_000_000_000


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_native_checkpoint_mapping_loads_all_text_parameters(tmp_path, tied, sharded):
    config = build_model_config(
        "debugmodel", parallelism=ParallelismConfig(), seq_len=128
    )
    config.vision_encoder = None
    config.vocab_size = 32
    config.tok_embeddings.num_embeddings = 32
    config.lm_head.out_features = 32
    config.enable_weight_tying = tied
    source = config.build()
    source.init_weights(buffer_device=torch.device("cpu"))
    adapter = source.state_dict_adapter_cls(config, str(tmp_path))
    hf = adapter.to_hf(dict(source.named_parameters()))
    # Simulate the multimodal HF checkpoint naming while training text only.
    hf = {
        key.replace("model.", "model.language_model.", 1)
        if key.startswith("model.")
        else key: value.detach().contiguous()
        for key, value in hf.items()
    }
    if sharded:
        # Force gate and up into separate shards.
        first = {key: value for key, value in hf.items() if ".up_proj." not in key}
        second = {key: value for key, value in hf.items() if ".up_proj." in key}
        save_file(first, tmp_path / "first.safetensors")
        save_file(second, tmp_path / "second.safetensors")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        **dict.fromkeys(first, "first.safetensors"),
                        **dict.fromkeys(second, "second.safetensors"),
                    }
                }
            )
        )
    else:
        save_file(hf, tmp_path / "model.safetensors")
    destination = config.build()
    load_hf_weights(destination, config, tmp_path)
    for (name, before), (_, after) in zip(
        source.named_parameters(), destination.named_parameters(), strict=True
    ):
        torch.testing.assert_close(before, after, msg=name)


def test_fused_mlp_preserves_independent_gate_up_adapters_and_export():
    torch.manual_seed(7)
    base = Linear.Config(
        in_features=5, out_features=7, num_linears=2, inplace_wgrad_accum=False
    ).build()
    routing = AdapterBatch(("0", "1"))
    layer = MultiLoRALinear(base, routing, 4)
    generator = torch.Generator().manual_seed(3)
    for slot in routing.slots:
        layer.add_adapter(slot, 2, 8, generator)
        nn.init.normal_(layer.b[slot][:, :, :2])
    x = torch.randn(2, 3, 5, requires_grad=True)
    actual = layer(x.flatten(0, 1)).view(2, 3, 2, 7)
    references = []
    for i, slot in enumerate(routing.slots):
        weights = layer.peft_tensors("mlp.gate_up_proj", slot, 2)
        branches = []
        for branch, target in enumerate(("gate_proj", "up_proj")):
            a = weights[f"mlp.{target}.lora_A.weight"]
            b = weights[f"mlp.{target}.lora_B.weight"]
            branches.append(x[i] @ base.weight[branch].T + x[i] @ a.T @ b.T * 4)
        references.append(torch.stack(branches, dim=1))
    expected = torch.stack(references)
    torch.testing.assert_close(actual, expected)
    parameters = [x, *layer.a.values(), *layer.b.values()]
    actual_grads = torch.autograd.grad(
        actual.square().sum(), parameters, retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    for got, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, reference)
