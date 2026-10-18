from spindle import deploy_wizard


def test_bundled_presets_keep_one_config_per_model_mode_and_context():
    presets = deploy_wizard.bundled_presets()
    keys = [(p.model, p.mode, p.context) for p in presets]
    names = {p.name for p in presets}
    assert len(keys) == len(set(keys))
    assert "qwen35-9b-lora-16k" in names
    assert "qwen35-9b-lora-16k-single" not in names


def test_pick_uses_the_shortest_context_that_fits():
    presets = deploy_wizard.bundled_presets()
    model = "Qwen/Qwen3.5-9B-Base"
    assert (
        deploy_wizard.pick(presets, model, "lora", 16_384).name == "qwen35-9b-lora-16k"
    )
    assert (
        deploy_wizard.pick(presets, model, "lora", 32_768).name == "qwen35-9b-lora-64k"
    )
    assert (
        deploy_wizard.pick(presets, model, "lora", 262_144).name == "qwen35-9b-lora-64k"
    )


def test_models_are_listed_smallest_trainer_first():
    presets = deploy_wizard.bundled_presets()
    smallest = [
        min(p.trainer_gpu_count for p in presets if p.model == model)
        for model in deploy_wizard.models_by_size(presets)
    ]
    assert smallest == sorted(smallest)


def test_models_with_lora_and_full_offer_both_modes():
    presets = deploy_wizard.bundled_presets()
    assert deploy_wizard.modes_for(presets, "Qwen/Qwen3.5-4B") == ["lora", "full"]
    assert deploy_wizard.modes_for(presets, "Qwen/Qwen3.6-27B") == ["full"]
