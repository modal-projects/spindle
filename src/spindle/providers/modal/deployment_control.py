from collections.abc import Awaitable, Callable

from spindle.control_plane import ControlPlane


class DeploymentControlPlane(ControlPlane):
    def __init__(
        self,
        *args,
        request_trainer_reconciliation: Callable[..., Awaitable[object]],
        trainer_maximum_instances: Callable[[str], int],
        trainer_models_per_instance: Callable[[str], int],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.request_trainer_reconciliation = request_trainer_reconciliation
        self.trainer_maximum_instances = trainer_maximum_instances
        self.trainer_models_per_instance = trainer_models_per_instance

    async def _trainer_demand_changed(self, definition_id: str) -> None:
        await self._reconcile_trainers(definition_id)

    async def _reconcile_trainers(self, definition_id: str) -> bool | None:
        result = await self.request_trainer_reconciliation(
            definition_id,
            revision=None,
            minimum_instances=0,
            maximum_instances=self.trainer_maximum_instances(definition_id),
            models_per_instance=self.trainer_models_per_instance(definition_id),
            scale_up=True,
        )
        return result if isinstance(result, bool) else None

    def _trainer_saturation_is_error(self, _definition_id: str) -> bool:
        return True
