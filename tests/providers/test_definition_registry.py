from spindle.providers.modal.app import (
    DEFINITIONS,
    module_for,
)


def test_definition_registry_resolves_every_definition() -> None:
    for definition in DEFINITIONS:
        assert module_for(definition.definition_id) is definition
        assert definition.trainer_app_name
