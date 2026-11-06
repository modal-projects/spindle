import asyncio
import threading

import tinker

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
from tests.control_plane.test_sdk_e2e import (
    API_KEY,
    BASE_MODEL,
    DEFINITION,
    DEFINITIONS,
)
from tests.support import TinkerStubExecutor, serve


def test_stock_sdk_can_pipeline_independent_adapters(sdk_garbage_collected_off_loop):
    blocked, paired = threading.Event(), threading.Event()
    block_model = []

    class Executor(TinkerStubExecutor):
        async def execute_forward_backward_batch(self, commands):
            if commands[0].model_id in block_model:
                blocked.set()
                async with asyncio.timeout(10):
                    while (
                        sum(
                            len(model.buffered)
                            for server in platform._servers.values()
                            for model in server._models.values()
                        )
                        < 2
                    ):
                        await asyncio.sleep(0.01)
            return await super().execute_forward_backward_batch(commands)

        async def execute_optimizer_group(self, commands):
            assert commands[0].model_id != commands[1].model_id
            paired.set()
            return tuple(
                [
                    await self.execute(item.model_id, item.kind, item.payload)
                    for item in commands
                ]
            )

    platform = LocalEnginePlatform(DEFINITION, Executor, overlap_optimizer_step=True)
    plane = ControlPlane(InMemoryKeyValueStore(), platform)
    app = create_control_plane_app(
        plane, DEFINITIONS, api_key=API_KEY, retrieve_window=1
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key=API_KEY)
        a, b, blocker = [
            service.create_lora_training_client(base_model=BASE_MODEL, rank=32)
            for _ in range(3)
        ]
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints([1, 2]),
            loss_fn_inputs={"target_tokens": [2, 3], "weights": [1.0, 1.0]},
        )
        a.forward_backward([datum], "cross_entropy").result(timeout=10)
        block_model.append(blocker.model_id)
        pending = blocker.forward_backward([datum], "cross_entropy")
        assert blocked.wait(10)
        optimizer = a.optim_step(tinker.AdamParams(learning_rate=1e-4))
        backward = b.forward_backward([datum], "cross_entropy")
        assert optimizer.result(timeout=10).metrics == {"lr": 1e-4}
        assert backward.result(timeout=10).metrics["loss:sum"] == 1.25
        assert pending.result(timeout=10).metrics["loss:sum"] == 1.25
        assert paired.is_set()
