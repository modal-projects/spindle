import asyncio

import httpx
import pytest

from spindle.engine import Engine, FutureStatus, OperationKind
from spindle.engine.api import Command, command_failure
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.operations import parse_operation_payload
from spindle.engine.spmd import DistributedExecutor
from tests.engine.test_server import forward, forward_backward
from tests.engine.test_spmd import RecordingBackend
from tests.support import EchoExecutor


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [None, "group", "slot"])
def test_group_preserves_sequence_and_drains_before_close(enabled, failure):
    async def run():
        blocked, release = asyncio.Event(), asyncio.Event()
        grouped, finish = asyncio.Event(), asyncio.Event()
        calls = []

        class Executor(EchoExecutor):
            async def execute_forward_backward_batch(self, commands):
                calls.append(tuple(item.model_id for item in commands))
                if commands[0].model_id == "blocker":
                    blocked.set()
                    await release.wait()
                return await super().execute_forward_backward_batch(commands)

            async def execute_optimizer_group(self, commands):
                calls.append(tuple(item.model_id for item in commands))
                grouped.set()
                await finish.wait()
                if failure == "group":
                    raise RuntimeError("group execution failed")
                results = []
                for item in commands:
                    if failure == "slot" and item.model_id == "c":
                        results.append(command_failure("slot c failed"))
                    else:
                        results.append(
                            await self.execute(item.model_id, item.kind, item.payload)
                        )
                return tuple(results)

        server = Engine(Executor(), overlap_optimizer_step=enabled)
        for model in ("blocker", "a", "b", "c"):
            await server.accept_model(model, {})
        await forward_backward(server, 1, model_id="blocker")
        await asyncio.wait_for(blocked.wait(), 1)
        optimizer_a = await server.optim_step(
            {"model_id": "a", "seq_id": 1, "adam_params": {}}
        )
        backward_b = await forward_backward(server, 1, model_id="b")
        optimizer_c = await server.optim_step(
            {"model_id": "c", "seq_id": 1, "adam_params": {}}
        )
        next_a = await forward(server, 2, model_id="a")
        next_c = await forward(server, 2, model_id="c")
        release.set()
        if enabled:
            await asyncio.wait_for(grouped.wait(), 1)
            assert calls == [("blocker",), ("a", "c", "b")]
            for future in (optimizer_a, optimizer_c, backward_b, next_a, next_c):
                assert (
                    await server.retrieve_future(future)
                ).status == FutureStatus.PENDING
            closing = asyncio.create_task(server.close())
            await asyncio.sleep(0)
            assert not closing.done()
            finish.set()
            await asyncio.wait_for(closing, 1)
            group_status = (
                FutureStatus.FAILED if failure == "group" else FutureStatus.COMPLETE
            )
            assert (await server.retrieve_future(optimizer_a)).status == group_status
            assert (await server.retrieve_future(backward_b)).status == group_status
            state_c = await server.retrieve_future(optimizer_c)
            if failure == "slot":
                assert state_c.status == FutureStatus.FAILED
                assert state_c.error == "slot c failed"
            else:
                assert state_c.status == group_status
            assert calls.count(("a", "c", "b")) == 1
        else:
            for future in (next_a, next_c):
                assert (
                    await server.retrieve_future(future, timeout=1)
                ).status == FutureStatus.COMPLETE
            assert not grouped.is_set()
            assert ("a", "c", "b") not in calls
            await server.close()

    asyncio.run(run())


def test_group_http_dispatch_isolates_slot_failures_and_excludes_adapters():
    class Backend(RecordingBackend):
        def optim_step_group(self, optimizers, batch):
            outputs = tuple(
                "boom" if model_id == "bad" else self.optim_step((model_id,), adam)[0]
                for model_id, adam in optimizers
            )
            return outputs, (() if batch is None else self.forward_backward(batch))

    async def run():
        backend = Backend()
        executor = DistributedExecutor(backend)
        client = HttpBackendClient(
            "http://backend",
            transport=httpx.ASGITransport(app=create_backend_app(executor)),
        )

        def optimizer(model_id):
            return Command(
                model_id,
                OperationKind.OPTIM_STEP,
                parse_operation_payload(OperationKind.OPTIM_STEP, {"adam_params": {}}),
            )

        backward = Command(
            "b",
            OperationKind.FORWARD_BACKWARD,
            parse_operation_payload(
                OperationKind.FORWARD_BACKWARD, {"data": [], "loss_fn": "cross_entropy"}
            ),
        )
        results = await client.execute_optimizer_group(
            (optimizer("a"), optimizer("bad"), backward)
        )
        assert results[0] == {"metrics": {"grad_norm:mean": 0.5}}
        assert results[1] == command_failure("boom")
        assert results[2]["metrics"] == {"loss:mean": 1.0}
        results = await client.execute_optimizer_group((optimizer("a"), optimizer("c")))
        assert len(results) == 2
        with pytest.raises(ValueError, match="different adapters"):
            await client.execute_optimizer_group(
                (optimizer("a"), Command("a", backward.kind, backward.payload))
            )
        with pytest.raises(ValueError, match="duplicate"):
            await client.execute_optimizer_group((optimizer("a"), optimizer("a")))
        await client.close()

    asyncio.run(run())
