"""Select recipes in deployment order, or explicitly by definition ID."""

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, validate_frontend


def _settings(config):
    return {**config, "spindle_version": None}


class DeploymentRoutes:
    """Recipes deployed with the frontend, then recipes added to it later.

    Added recipes are only ever inserted, so lookups reread them on a miss.
    """

    def __init__(self, definitions, added=None):
        self.base = tuple(definitions)
        self.added_store = added
        self.added_rows = ()
        self.added = ()

    @property
    def definitions(self):
        base_names = {d.name for d in self.base}
        return self.base + tuple(d for d in self.added if d.name not in base_names)

    @property
    def pending(self):
        """Added recipes whose apps are still deploying or failed to deploy."""
        return tuple(row for row in self.added_rows if row["state"] != "ready")

    def get(self, definition_id):
        return next(
            (d for d in self.definitions if d.definition_id == definition_id), None
        )

    async def refresh(self):
        if self.added_store is None:
            return
        self.added_rows = tuple(await self.added_store.list())
        self.added = tuple(
            DeploymentConfig.model_validate(row["config"])
            for row in self.added_rows
            if row["state"] == "ready"
        )

    async def load(self, definition_id):
        if self.get(definition_id) is None:
            await self.refresh()
        return self.get(definition_id)

    async def find(self, model, mode=None):
        selected = self.select(model, mode)
        if selected is None:
            await self.refresh()
            selected = self.select(model, mode)
        return selected

    async def add(self, name, overrides, base=None, recipe=None):
        """Derive a recipe from a deployed one, or take a full recipe, and start
        deploying it under this frontend's platform settings; returns its state."""
        if self.added_store is None:
            raise ValueError("this frontend does not accept added deployments")
        await self.refresh()
        if recipe is None:
            parent = self.get(base)
            if parent is None:
                raise ValueError(f"unknown base deployment {base!r}")
            recipe = vars(parent.recipe)
        recipe = {**recipe, "platform": self.base[0].recipe.platform}
        deployment = DeploymentConfig.create(
            BaseConfig(**recipe, overrides={**overrides, "name": name})
        )
        current = self.get(name)
        if current is None:
            validate_frontend(
                [d.recipe for d in self.definitions] + [deployment.recipe]
            )
            row = await self.added_store.add(deployment)
        else:
            row = {"config": current.model_dump(mode="json"), "state": "ready"}
        if _settings(row["config"]) != _settings(deployment.model_dump(mode="json")):
            raise ValueError(
                f"deployment {name!r} already exists with different settings"
            )
        return deployment, row["state"]

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
