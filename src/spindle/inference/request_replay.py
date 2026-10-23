"""Share one generation across duplicate deliveries of the same request ID."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi.responses import JSONResponse


@dataclass
class Generation:
    fingerprint: bytes
    task: asyncio.Task
    waiters: int = 0
    disconnect_timer: asyncio.Task | None = None


class RequestReplay:
    def __init__(
        self,
        *,
        ttl=30.0,
        disconnect_grace=1.0,
        max_entries=512,
        max_bytes=64 * 1024 * 1024,
    ):
        self.ttl = ttl
        self.disconnect_grace = disconnect_grace
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.pending = {}
        self.completed = OrderedDict()
        self.completed_bytes = 0
        self.closed = False

    def _prune(self):
        now = time.monotonic()
        while self.completed:
            _, (_, response, expires) = next(iter(self.completed.items()))
            if (
                expires > now
                and len(self.completed) <= self.max_entries
                and self.completed_bytes <= self.max_bytes
            ):
                break
            self.completed.popitem(last=False)
            self.completed_bytes -= len(response.body)

    def _finished(self, rid, generation):
        if self.pending.get(rid) is generation:
            del self.pending[rid]
        if generation.disconnect_timer is not None:
            generation.disconnect_timer.cancel()
        if generation.task.cancelled():
            response = JSONResponse(
                {"error": "Request attempt was cancelled; use a new request ID"},
                status_code=409,
                headers={"x-spindle-replay-state": "cancelled"},
            )
        elif generation.task.exception() is not None:
            return
        else:
            response = generation.task.result()
        # Non-string keys are unique batch pass-throughs, with no reusable reply.
        if (
            not self.closed
            and isinstance(rid, str)
            and len(response.body) <= self.max_bytes
        ):
            self.completed[rid] = (
                generation.fingerprint,
                response,
                time.monotonic() + self.ttl,
            )
            self.completed_bytes += len(response.body)
            self._prune()

    async def _abort_disconnected(self, generation):
        await asyncio.sleep(self.disconnect_grace)
        if generation.waiters == 0 and not generation.task.done():
            generation.task.cancel()
            await asyncio.gather(generation.task, return_exceptions=True)

    async def run(self, rid, payload, generate, is_disconnected):
        self._prune()
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        completed = self.completed.get(rid)
        generation = self.pending.get(rid)
        previous = (
            completed[0]
            if completed
            else (generation.fingerprint if generation else None)
        )
        if previous is not None and previous != fingerprint:
            return JSONResponse(
                {"error": "Request ID reused with a different payload"}, status_code=409
            )
        if completed is not None:
            return completed[1]
        if generation is None:
            generation = Generation(fingerprint, asyncio.create_task(generate()))
            self.pending[rid] = generation
            generation.task.add_done_callback(lambda _: self._finished(rid, generation))
        elif generation.task.cancelling():
            return JSONResponse(
                {"error": "Request attempt is cancelling; use a new request ID"},
                status_code=409,
                headers={"x-spindle-replay-state": "cancelling"},
            )
        if generation.disconnect_timer is not None:
            generation.disconnect_timer.cancel()
            generation.disconnect_timer = None
        generation.waiters += 1
        disconnected = False
        try:
            while not generation.task.done():
                done, _ = await asyncio.wait((generation.task,), timeout=0.25)
                if done:
                    break
                if await is_disconnected():
                    disconnected = True
                    return None
            return generation.task.result()
        finally:
            generation.waiters -= 1
            if generation.waiters == 0 and not generation.task.done():
                if disconnected:
                    # A reconnect may deliver its replay just after the old
                    # connection closes. Only this bounded interval retains work.
                    generation.disconnect_timer = asyncio.create_task(
                        self._abort_disconnected(generation)
                    )
                else:
                    # Explicit cancellation/shutdown does not need a grace period.
                    generation.task.cancel()
                    await asyncio.gather(generation.task, return_exceptions=True)

    async def aclose(self):
        self.closed = True
        tasks = []
        for generation in list(self.pending.values()):
            if generation.disconnect_timer is not None:
                generation.disconnect_timer.cancel()
                tasks.append(generation.disconnect_timer)
            generation.task.cancel()
            tasks.append(generation.task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.completed.clear()
        self.completed_bytes = 0
