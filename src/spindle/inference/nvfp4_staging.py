"""Bind online NVFP4 loaders to the isolated staging layer.

The online loader is a closure: it writes the supplied weight parameter but
looks up scale parameters on its captured layer. Copying that closure into a
CPU shadow therefore mutates live scales and leaves the staged scales stale.
Keep the original bound loader on the module so SGLang can rebind its owner,
then create fresh closures for each complete staging pass.
"""


def restore_nvfp4_loaders(layer):
    original = layer._nvfp4_staging_source_loader
    if getattr(original, "__self__", None) is not layer:
        raise RuntimeError(
            "NVFP4 staging source loader still references the live layer"
        )
    loader = layer.quant_method.get_online_weight_loader(layer, original)
    for parameter in layer.parameters(recurse=False):
        if hasattr(parameter, "_weight_loader"):
            parameter._weight_loader = loader
