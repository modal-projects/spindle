import torch

from spindle.inference.fp8_staging import (
    quantize_blocks,
    remember_loader_state,
    restore_loader_state,
)


def test_online_fp8_reload_restores_checkpoint_layout_and_loader_twice():
    class Weight(torch.nn.Parameter):
        pass

    layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    layer.weight = Weight(layer.weight.data, requires_grad=False)

    def load(parameter, values):
        assert parameter.shape == values.shape
        parameter.data.copy_(values)

    layer.weight.weight_loader = load
    remember_loader_state(layer, ("weight",))
    for scale in (2, 3):
        # Online FP8 creates a transposed plain Parameter with no loader attrs.
        layer.weight = torch.nn.Parameter(
            layer.weight.data.t().to(torch.float8_e4m3fn), requires_grad=False
        )
        remember_loader_state(layer, ("weight",))
        restore_loader_state(layer)
        assert isinstance(layer.weight, Weight)
        assert layer.weight.shape == (2, 3)
        assert layer.weight.dtype == torch.bfloat16
        layer.weight.weight_loader(
            layer.weight, torch.full((2, 3), scale, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            layer.weight, torch.full((2, 3), scale, dtype=torch.bfloat16)
        )


def test_block_fp8_matches_independent_expert_and_block_reference():
    torch.manual_seed(42)
    weights = torch.randn(2, 256, 384, dtype=torch.bfloat16)
    weights[0, :128, :128] = 0
    weights[1, 128:, 128:256] *= 100
    quantized, scales = quantize_blocks(weights)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 2, 3)
    for expert in range(2):
        for row in range(2):
            for column in range(3):
                block = weights[
                    expert,
                    row * 128 : (row + 1) * 128,
                    column * 128 : (column + 1) * 128,
                ].float()
                scale = block.abs().max().clamp_min(1e-12) / 448.0
                expected = (block / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                actual = quantized[
                    expert,
                    row * 128 : (row + 1) * 128,
                    column * 128 : (column + 1) * 128,
                ]
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=0, atol=0
                )
                torch.testing.assert_close(scales[expert, row, column], scale)
