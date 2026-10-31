import asyncio
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parents[2] / "examples" / "heterogeneous-samplers")
)

from readiness import SamplerGate


async def hang():
    await asyncio.Event().wait()


async def ready(value, delay=0):
    await asyncio.sleep(delay)
    return value


def test_rollout_starts_when_the_first_sampler_is_ready():
    gate = SamplerGate()

    async def scenario():
        await gate.open({"slow": hang(), "fast": ready("fast", 0.01)})
        assert set(gate.ready) == {"fast"}
        assert gate.pending_names() == ["slow"]
        adopting = asyncio.create_task(gate.adopt())
        await asyncio.sleep(0.05)
        adopting.cancel()
        await asyncio.gather(adopting, return_exceptions=True)
        assert gate.pending_names() == []
        assert "slow" not in gate.ready

    asyncio.run(scenario())


def test_a_later_sampler_can_join_before_training_finishes():
    gate = SamplerGate()

    async def scenario():
        await gate.open({"later": ready("later", 0.05), "first": ready("first")})
        assert set(gate.ready) == {"first"}
        adopting = asyncio.create_task(gate.adopt())
        await asyncio.sleep(0.1)
        adopting.cancel()
        await asyncio.gather(adopting, return_exceptions=True)
        assert gate.ready["later"] == "later"

    asyncio.run(scenario())
