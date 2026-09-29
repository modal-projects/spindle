from spindle.providers.modal.app import (
    DEFINITIONS,
    module_for,
    parameterization_for,
)


def test_definition_registry_resolves_every_definition() -> None:
    for definition in DEFINITIONS:
        assert module_for(definition.definition_id) is definition
        assert parameterization_for(definition.definition_id) == (
            definition.parameterization
        )
        assert definition.trainer_app_name
