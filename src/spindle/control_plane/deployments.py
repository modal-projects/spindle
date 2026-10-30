"""Select the longest-context recipe for a model, or a recipe by definition ID."""


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
        matches = [d for d in candidates if d.model == model]
        return max(matches, key=lambda d: d.max_context_length, default=None)

    def capabilities(self):
        rows = []
        for model in dict.fromkeys(d.model for d in self.definitions):
            recipes = [d for d in self.definitions if d.model == model]
            context = min(
                self.select(model, d.parameterization).max_context_length
                for d in recipes
            )
            rows.append({"model_name": model, "max_context_length": context})
            if len(recipes) > 1:
                rows += [
                    {
                        "model_name": d.definition_id,
                        "max_context_length": d.max_context_length,
                    }
                    for d in recipes
                ]
        return rows
