import pytest
import torch

from spindle.inference.nvfp4_staging import restore_nvfp4_loaders


class OnlineQuantizer:
    def get_online_weight_loader(self, layer, original):
        def load(parameter, values):
            original(parameter, values)
            # Like NVFP4's closure, the scale destination comes from the layer,
            # independently of the weight parameter supplied by the model loader.
            original(layer.scale, values.abs().max().reshape(1))

        return load


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.quant_method = OnlineQuantizer()
        self._nvfp4_staging_source_loader = self.load
        loader = self.quant_method.get_online_weight_loader(self, self.load)
        self.weight._weight_loader = loader
        self.scale._weight_loader = loader

    def load(self, parameter, values):
        assert parameter is self.weight or parameter is self.scale
        parameter.data.copy_(values)


def test_nvfp4_staging_updates_private_weights_and_scales_on_repeated_loads():
    live, shadow = Expert(), Expert()
    # The module's bound method was rebound, but a copied closure still captures
    # the live layer. This is the pinned SGLang cloning contract at restoration.
    shadow.weight._weight_loader = live.weight._weight_loader
    shadow.scale._weight_loader = live.scale._weight_loader
    for maximum in (2.0, 3.0):
        restore_nvfp4_loaders(shadow)
        values = torch.tensor([maximum, -maximum, 0.5, 0.0])
        shadow.weight._weight_loader(shadow.weight, values)
        torch.testing.assert_close(shadow.weight, values)
        torch.testing.assert_close(shadow.scale, torch.tensor([maximum]))
        torch.testing.assert_close(live.weight, torch.ones(4))
        torch.testing.assert_close(live.scale, torch.ones(1))
        assert shadow.scale._weight_loader is shadow.weight._weight_loader


def test_nvfp4_staging_rejects_a_source_loader_owned_by_the_live_layer():
    live, shadow = Expert(), Expert()
    shadow._nvfp4_staging_source_loader = live.load
    with pytest.raises(RuntimeError, match="still references the live layer"):
        restore_nvfp4_loaders(shadow)
