from spindle.providers.modal.app import (
    DEPLOYMENTS,
    module_for,
    parameterization_for,
)


def test_definition_registry_resolves_every_definition() -> None:
    for definition in DEPLOYMENTS.base:
        assert module_for(definition.definition_id) is definition
        assert parameterization_for(definition.definition_id) == (
            definition.parameterization
        )
        assert definition.trainer_app_name
