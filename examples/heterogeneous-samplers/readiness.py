"""Start rollout when the first sampler is up, and stop waiting when training ends."""

import asyncio


class SamplerGate:
    def __init__(self):
        self.ready = {}
        self.failures = {}
        self._pending = {}

    def pending_names(self):
        return [self._pending[task] for task in self._pending]

    async def open(self, checks):
        self._pending = {
            asyncio.create_task(check): name for name, check in checks.items()
        }
        try:
            while self._pending and not self.ready:
                done, _ = await asyncio.wait(
                    set(self._pending), return_when=asyncio.FIRST_COMPLETED
                )
                self._take(done)
            if not self.ready:
                raise RuntimeError(f"no samplers became ready: {self.failures}")
        except BaseException:
            await self.cancel()
            raise

    async def adopt(self):
        """Keep accepting later samplers until cancelled."""
        try:
            while self._pending:
                done, _ = await asyncio.wait(
                    set(self._pending), return_when=asyncio.FIRST_COMPLETED
                )
                self._take(done)
        finally:
            await self.cancel()

    async def cancel(self):
        tasks = list(self._pending)
        for task in tasks:
            task.cancel()
        self._pending.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _take(self, done):
        for task in done:
            name = self._pending.pop(task)
            try:
                self.ready[name] = task.result()
            except Exception as exc:
                self.failures[name] = exc
