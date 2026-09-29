"""Select recipes in deployment order, or explicitly by definition ID."""


class DeploymentRoutes:
    def __init__(self, definitions):
        self.definitions = tuple(definitions)

    def select(self, model, mode=None):
        candidates = [
            d for d in self.definitions if mode is None or d.parameterization == mode
        ]
        for definition in candidates:
            if definition.definition_id == model:
                return definition
        for definition in candidates:
            if definition.model == model:
                return definition
        return None

    def capabilities(self):
        selected = {}
        for definition in self.definitions:
            selected.setdefault(
                (definition.model, definition.parameterization), definition
            )
        contexts = {}
        for (model, _), definition in selected.items():
            contexts[model] = min(
                contexts.get(model, definition.max_context_length),
                definition.max_context_length,
            )
        return [
            {"model_name": model, "max_context_length": context}
            for model, context in contexts.items()
        ]
