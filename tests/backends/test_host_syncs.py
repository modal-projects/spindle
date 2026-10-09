from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from spindle.backends.miles_runtime import host_syncs  # noqa: E402

N_ADAPTERS, HIDDEN, OUT, RANK, EXPERTS = 4, 16, 12, 4, 3


def _upstream_dense_mm(x, stacked_weights, *, token_splits, offsets):
    """CPU form of Megatron-Bridge's ``_dense_multi_lora_mm`` (its per-slot path)."""
    inputs = x.split(token_splits, dim=0)
    return torch.cat([F.linear(i, w) for i, w in zip(inputs, stacked_weights)])


def _layers():
    def co_permute(dispatcher, slot_ids, num_local_experts):
        return slot_ids.index_select(
            0, dispatcher.reversed_local_input_permutation_mapping
        )

    return SimpleNamespace(
        parallel_state=SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
        ),
        _dense_multi_lora_mm=_upstream_dense_mm,
        _co_permute_slot_ids=co_permute,
    )


def _adapter(seed):
    generator = torch.Generator().manual_seed(seed)
    weight_in = torch.randn(RANK, HIDDEN, generator=generator).requires_grad_()
    weight_out = torch.randn(OUT, RANK, generator=generator).requires_grad_()
    return SimpleNamespace(
        linear_in=SimpleNamespace(weight=weight_in),
        linear_out=SimpleNamespace(weight=weight_out),
    )


def _dense(token_splits, alphas):
    layer = SimpleNamespace(
        adapters=[_adapter(seed) for seed in range(N_ADAPTERS)],
        alpha_values=torch.tensor(alphas, dtype=torch.float32),
        rank_values=torch.full((N_ADAPTERS,), float(RANK)),
        tokens_per_adapter=torch.tensor(token_splits, dtype=torch.int32),
        tokens_per_adapter_splits=tuple(token_splits),
        tokens_per_adapter_total=sum(token_splits),
        _adapter_enabled=True,
        disable_sequence_parallel_comm=True,
        input_is_parallel=False,
        replicate_adapter=True,
        _external_output_reduce=False,
        _gather_output=False,
        use_a2a=False,
        base_linear_name="test",
    )
    layer.base_linear_forward = lambda x: (torch.zeros(x.shape[0], OUT), None, x)
    for slot, alpha in enumerate(alphas):
        host_syncs._unit_scaling(layer)[slot] = alpha == RANK
    return layer


def _upstream_dense(layer, x):
    offsets = layer.tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)
    splits = layer.tokens_per_adapter_splits
    stacked_A = torch.stack([a.linear_in.weight for a in layer.adapters])
    stacked_B = torch.stack([a.linear_out.weight for a in layer.adapters])
    mid = _upstream_dense_mm(x, stacked_A, token_splits=splits, offsets=offsets)
    out = _upstream_dense_mm(mid, stacked_B, token_splits=splits, offsets=offsets)
    scaling = layer.alpha_values / layer.rank_values
    per_token = torch.repeat_interleave(scaling, layer.tokens_per_adapter)
    return out * per_token.unsqueeze(-1)


def _weights(layer):
    return [
        weight
        for adapter in layer.adapters
        for weight in (adapter.linear_in.weight, adapter.linear_out.weight)
    ]


@pytest.mark.parametrize(
    ("token_splits", "alphas"),
    [
        ((0, 7, 0, 0), (RANK, RANK, 1.0, RANK)),
        ((0, 7, 0, 0), (RANK, 2.0, RANK, RANK)),
        ((3, 0, 5, 1), (RANK, RANK, RANK, RANK)),
        ((3, 0, 5, 1), (RANK, 1.0, 8.0, 2.0)),
    ],
)
def test_dense_forward_matches_upstream(token_splits, alphas):
    layer = _dense(token_splits, alphas)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(sum(token_splits), HIDDEN, generator=generator)
    out, _ = host_syncs._dense_forward(_layers())(layer, x)
    expected = _upstream_dense(layer, x)
    assert torch.equal(out, expected)

    grads = torch.autograd.grad(out.square().sum(), _weights(layer))
    expected_grads = torch.autograd.grad(expected.square().sum(), _weights(layer))
    for grad, expected_grad in zip(grads, expected_grads, strict=True):
        assert torch.equal(grad, expected_grad)


def _dispatch(token_splits, topk=2, seed=3):
    """Expert-major rows for ``sum(token_splits)`` tokens routed to ``topk`` experts."""
    tokens = sum(token_splits)
    generator = torch.Generator().manual_seed(seed)
    choices = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:topk] for _ in range(tokens)]
    )
    expert_of_row = choices.reshape(-1)
    token_of_row = torch.arange(tokens).repeat_interleave(topk)
    order = torch.argsort(expert_of_row, stable=True)
    dispatcher = SimpleNamespace(
        ep_size=1,
        tp_size=1,
        hidden_shape_before_permute=(tokens, HIDDEN),
        reversed_local_input_permutation_mapping=token_of_row[order],
    )
    return dispatcher, torch.bincount(expert_of_row, minlength=EXPERTS)


def _upstream_routing(dispatcher, tokens_per_expert, tokens_per_adapter):
    slot_ids = torch.repeat_interleave(
        torch.arange(N_ADAPTERS, dtype=torch.int32), tokens_per_adapter
    )
    slot_ids = slot_ids.index_select(
        0, dispatcher.reversed_local_input_permutation_mapping
    )
    expert_ids = torch.repeat_interleave(torch.arange(EXPERTS), tokens_per_expert)
    keys = slot_ids.long() * EXPERTS + expert_ids
    counts = torch.bincount(keys, minlength=N_ADAPTERS * EXPERTS)
    return (
        torch.argsort(keys, stable=True),
        counts.cumsum(dim=0, dtype=torch.int32),
        counts.view(N_ADAPTERS, EXPERTS).sum(dim=1),
    )


@pytest.mark.parametrize("token_splits", [(0, 9, 0, 0), (9, 0, 0, 0), (4, 0, 3, 2)])
def test_expert_routing_matches_upstream(token_splits):
    dispatcher, tokens_per_expert = _dispatch(token_splits)
    reference = SimpleNamespace(
        n_adapters=N_ADAPTERS,
        num_local_experts=EXPERTS,
        tokens_per_adapter=torch.tensor(token_splits, dtype=torch.int32),
        tokens_per_adapter_splits=tuple(token_splits),
    )
    routing = host_syncs.build_routing(
        _layers(), dispatcher, tokens_per_expert, reference, torch.device("cpu")
    )
    sort_idx, offsets, slot_counts = _upstream_routing(
        dispatcher, tokens_per_expert, reference.tokens_per_adapter
    )

    single_adapter = sum(1 for count in token_splits if count) == 1
    assert (routing.sort_idx is None) == single_adapter
    if single_adapter:
        assert torch.equal(sort_idx, torch.arange(sort_idx.numel()))
    else:
        assert torch.equal(routing.sort_idx, sort_idx)
        assert torch.equal(
            routing.inverse_idx[sort_idx], torch.arange(sort_idx.numel())
        )
    assert torch.equal(routing.group_offsets, offsets)
    assert torch.equal(routing.slot_token_counts, slot_counts)
    assert routing.num_tokens == sort_idx.numel()


def test_expert_parallel_routing_does_not_assume_local_adapters():
    dispatcher, tokens_per_expert = _dispatch((0, 9, 0, 0))
    dispatcher.ep_size = 2
    reference = SimpleNamespace(
        n_adapters=N_ADAPTERS,
        num_local_experts=EXPERTS,
        tokens_per_adapter=torch.tensor((0, 9, 0, 0), dtype=torch.int32),
        tokens_per_adapter_splits=(0, 9, 0, 0),
    )
    routing = host_syncs.build_routing(
        _layers(), dispatcher, tokens_per_expert, reference, torch.device("cpu")
    )
    assert routing.sort_idx is not None
    assert routing.active_slots is None


def _fake_layers():
    class MultiLoRALinear:
        def __init__(self):
            self.alpha_values = torch.ones(N_ADAPTERS)
            self.rank_values = torch.full((N_ADAPTERS,), float(RANK))

        def init_adapter_slot(self, idx, rank, alpha):
            self.alpha_values[idx] = alpha
            self.rank_values[idx] = rank

        def clear_adapter_slot(self, idx):
            self.alpha_values[idx] = 0
            self.rank_values[idx] = RANK

    class MultiLoRAGroupedExpertLinear(MultiLoRALinear):
        pass

    return SimpleNamespace(
        **vars(_layers()),
        MultiLoRALinear=MultiLoRALinear,
        MultiLoRAGroupedExpertLinear=MultiLoRAGroupedExpertLinear,
        _make_slot_routing_hook=None,
    )


def test_slot_lifecycle_tracks_unit_scaling():
    layers = _fake_layers()
    assert host_syncs.install_multi_lora(layers)
    assert host_syncs.install_multi_lora(layers)
    layer = layers.MultiLoRAGroupedExpertLinear()
    layer.init_adapter_slot(0, RANK, float(RANK))
    layer.init_adapter_slot(1, RANK, 2.0 * RANK)
    assert host_syncs._unit_scaling(layer) == {0: True, 1: False}
    layer.clear_adapter_slot(0)
    assert host_syncs._unit_scaling(layer) == {1: False}


def test_install_skips_unpinned_bridge_and_honors_flag(monkeypatch):
    monkeypatch.setattr(host_syncs, "bridge_commit", lambda: "0" * 40)
    assert host_syncs.install_multi_lora() is False
    monkeypatch.setenv(host_syncs.ENV_FLAG, "0")
    assert host_syncs.install() == {"multi_lora": False, "geglu": False}


def _fused():
    calls = []

    def apply(input, weights, fp8_input_store, offset):
        calls.append(offset)
        return (input[:, : input.shape[1] // 2] + offset) * weights

    def weighted_bias_quick_geglu_impl(
        input, bias, weights, fp8_input_store=False, linear_offset=0.0, clamp_value=None
    ):
        return None

    return SimpleNamespace(
        WeightedQuickGeGLUFunction=SimpleNamespace(apply=apply),
        WeightedBiasQuickGeGLUFunction=None,
        weighted_bias_quick_geglu_impl=weighted_bias_quick_geglu_impl,
    ), calls


def test_geglu_reuses_offset_tensor():
    fused, calls = _fused()
    geglu = host_syncs._geglu(fused)
    x, w = torch.randn(5, 8), torch.rand(5, 1)
    first = geglu(x, None, w, False, 1.0, 7.0)
    second = geglu(x, None, w, False, 1.0, 7.0)
    assert torch.equal(first, second)
    assert calls[0] is calls[1]
    assert calls[0].item() == 1.0
    geglu(x.view(5, 1, 8), None, w, False, 0.0, None)
    assert calls[2].item() == 0.0


def test_geglu_patch_requires_known_source():
    fused, _ = _fused()
    original = fused.weighted_bias_quick_geglu_impl
    assert host_syncs.install_geglu(fused, experts=None) is False
    assert fused.weighted_bias_quick_geglu_impl is original
