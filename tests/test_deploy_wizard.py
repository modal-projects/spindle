import modal
import pytest
from rich.console import Console

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


def test_models_are_grouped_by_provider_smallest_trainer_first():
    presets = deploy_wizard.bundled_presets()
    models = deploy_wizard.models_by_provider(presets)
    providers = list(dict.fromkeys(deploy_wizard.provider(model) for model in models))
    assert providers == list(deploy_wizard.PROVIDER_ORDER)
    for name in providers:
        smallest = [
            min(p.trainer_gpu_count for p in presets if p.model == model)
            for model in models
            if deploy_wizard.provider(model) == name
        ]
        assert smallest == sorted(smallest)


def test_default_selection_deploys_one_qwen_lora():
    presets = deploy_wizard.bundled_presets()
    assert deploy_wizard.modes_for(presets, deploy_wizard.DEFAULT_MODEL)[0] == "lora"
    preset = deploy_wizard.pick(
        presets, deploy_wizard.DEFAULT_MODEL, "lora", deploy_wizard.DEFAULT_CONTEXT
    )
    assert preset.name == "qwen36-35b-a3b-lora-32k"


def test_models_with_lora_and_full_offer_both_modes():
    presets = deploy_wizard.bundled_presets()
    assert deploy_wizard.modes_for(presets, "Qwen/Qwen3.5-4B") == ["lora", "full"]
    assert deploy_wizard.modes_for(presets, "Qwen/Qwen3.6-27B") == ["full"]


def test_setup_check_lists_every_missing_secret_with_a_fix(monkeypatch):
    present = {"spindle-api": {"TINKER_API_KEY"}, "huggingface-secret": {"PLACEHOLDER"}}

    class FakeSecret:
        def __init__(self, name, keys):
            self.name, self.keys = name, keys

        def hydrate(self):
            if self.name not in present:
                raise modal.exception.NotFoundError("missing")
            missing = set(self.keys) - present[self.name]
            if missing:
                raise modal.exception.ConflictError(
                    f"Secret is missing key(s): {', '.join(sorted(missing))}"
                )

    monkeypatch.setattr(
        modal.Secret,
        "from_name",
        lambda name, environment_name=None, required_keys=(): FakeSecret(
            name, required_keys
        ),
    )
    console = Console(record=True, width=160)
    with pytest.raises(deploy_wizard.SetupIncomplete):
        deploy_wizard.check_setup(console, "dev")
    output = console.export_text()
    assert "spindle-proxy · MODAL_PROXY_TOKEN_ID, MODAL_PROXY_TOKEN_SECRET" in output
    assert "(not found)" in output
    assert "missing key(s): HF_TOKEN" in output
    assert "modal secret create huggingface-secret HF_TOKEN=hf_..." in output
    assert "modal secret create spindle-api" not in output


def test_setup_check_passes_when_every_secret_exists(monkeypatch):
    class PresentSecret:
        def hydrate(self):
            pass

    monkeypatch.setattr(modal.Secret, "from_name", lambda *a, **k: PresentSecret())
    console = Console(record=True, width=160)
    deploy_wizard.check_setup(console, None)
    assert "✗" not in console.export_text()
