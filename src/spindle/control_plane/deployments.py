"""Select recipes in deployment order, or explicitly by definition ID."""

import time

from spindle.deployments import validate_frontend

REFRESH_INTERVAL_S = 5.0


class DeploymentRoutes:
    """Recipes deployed with the frontend, then recipes added to it later."""

    def __init__(self, definitions, added=None, clock=time.monotonic):
        self.base = tuple(definitions)
        self.added_store = added
        self.added = ()
        self.refreshed_at = None
        self.clock = clock

    @property
    def definitions(self):
        base_names = {d.name for d in self.base}
        return self.base + tuple(d for d in self.added if d.name not in base_names)

    def get(self, definition_id):
        return next(
            (d for d in self.definitions if d.definition_id == definition_id), None
        )

    async def refresh(self, max_age=REFRESH_INTERVAL_S):
        if self.added_store is None:
            return
        if self.refreshed_at is not None and self.clock() - self.refreshed_at < max_age:
            return
        self.added = tuple(await self.added_store.list())
        self.refreshed_at = self.clock()

    async def load(self, definition_id):
        """The recipe for definition_id, rereading added recipes if it is unknown."""
        if self.get(definition_id) is None:
            await self.refresh(max_age=0)
        return self.get(definition_id)

    async def find(self, model, mode=None):
        selected = self.select(model, mode)
        if selected is None:
            await self.refresh(max_age=0)
            selected = self.select(model, mode)
        return selected

    async def add(self, deployment):
        if self.added_store is None:
            raise ValueError("this frontend does not accept added deployments")
        await self.refresh(max_age=0)
        current = self.get(deployment.name)
        if current is None:
            validate_frontend(
                [d.recipe for d in self.definitions] + [deployment.recipe]
            )
            current = await self.added_store.add(deployment)
            await self.refresh(max_age=0)
        if current.model_dump(mode="json") != deployment.model_dump(mode="json"):
            raise ValueError(
                f"deployment {deployment.name!r} already exists with different settings"
            )
        return current

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
