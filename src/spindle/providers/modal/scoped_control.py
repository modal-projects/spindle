"""Retry-safe model preparation for scoped deployments."""

from spindle.control_plane import ControlPlane
from spindle.control_plane.trainer_reconciler import reconcile_trainers


class ScopedControlPlane(ControlPlane):
    def __init__(self, *args, prepare_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_scoped_model = prepare_model

    async def create_model(self, **kwargs):
        creation = await super().create_model(**kwargs)
        # Creation is durable before infrastructure preparation. A retried API
        # request must finish preparation, not silently skip a failed first try.
        await self.prepare_scoped_model(creation.model)
        return creation

    async def _trainer_demand_changed(self, definition_id: str) -> None:
        await self._reconcile_trainers(definition_id)

    async def _reconcile_trainers(self, definition_id: str) -> None:
        await reconcile_trainers(
            self.kv,
            self.engines,
            definition_id,
            revision=None,
            minimum_instances=0,
            maximum_instances=1,
            models_per_instance=1,
        )
