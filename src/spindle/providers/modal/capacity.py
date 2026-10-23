import time
from collections.abc import Awaitable, Callable

import modal

CHECK_INTERVAL_S = 10.0


def gpu_label(gpu: str | list[str], gpus_per_node: int, nodes: int = 1) -> str:
    names = [gpu] if isinstance(gpu, str) else list(gpu)
    per_node = f"{gpus_per_node}× {' or '.join(names)}"
    return per_node if nodes == 1 else f"{nodes} nodes of {per_node}"


def gpu_wait(
    function_for: Callable[[str], modal.Function],
    label_for: Callable[[str], str],
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[str], Awaitable[str | None]]:
    """Report the requested GPUs while a function has queued inputs and no containers.

    Modal counts a container from placement onward, so a backlog with zero
    containers means the request is waiting for GPU capacity, not booting.
    """
    checked: dict[str, tuple[float, str | None]] = {}

    async def wait(definition_id: str) -> str | None:
        cached = checked.get(definition_id)
        if cached is not None and clock() - cached[0] < CHECK_INTERVAL_S:
            return cached[1]
        stats = await function_for(definition_id).get_current_stats.aio()
        queued = stats.backlog > 0 and stats.num_total_runners == 0
        result = label_for(definition_id) if queued else None
        checked[definition_id] = (clock(), result)
        return result

    return wait
