"""Replica-owned admission, fair batching, and exact-restore merge transitions.

No cluster coordinator participates in admission. A single asyncio task owns
each engine's mode; request handlers only append to a bounded local queue.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class MergeProfile:
    # Measured on the deployed model, rank, hardware and batch size. Seconds.
    unmerged_token_seconds: float
    merged_token_seconds: float
    prefill_tokens_per_second: float
    merge_seconds: float
    unmerge_seconds: float
    capacity: int
    queue_buffer: int = 4
    max_wait_seconds: float = 5.0
    enabled: bool = True
    initial_merge_seconds: float | None = None

    def __post_init__(self):
        if self.capacity < 1 or self.queue_buffer < 1:
            raise ValueError("Admission capacity and queue buffer must be positive")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.unmerged_token_seconds,
                self.merged_token_seconds,
                self.prefill_tokens_per_second,
                self.merge_seconds,
                self.unmerge_seconds,
                self.max_wait_seconds,
            )
        ):
            raise ValueError("Merge service measurements must be positive")
        if self.initial_merge_seconds is not None and (
            not math.isfinite(self.initial_merge_seconds)
            or self.initial_merge_seconds <= 0
        ):
            raise ValueError("Initial merge measurement must be positive")


class ReplicaBusy(Exception):
    pass


@dataclass
class Pending:
    adapter: str
    prompt_tokens: int
    output_tokens: int
    arrived: float
    prepare: object
    execute: object
    future: asyncio.Future


class ReplicaScheduler:
    def __init__(self, client, profile, *, invalidate):
        self.client, self.profile, self.invalidate = client, profile, invalidate
        self.pending = []
        self.running = {}
        self.mode = None
        self.merge_initialized = False
        self.transitioning = False
        self.failure = None
        self.changed = asyncio.Event()
        self.events = []
        self.completed = Counter()
        self.transitions = Counter()
        self.rejections = 0
        self.worker = asyncio.create_task(self._run())

    @property
    def active(self):
        return len(self.pending) + len(self.running)

    def snapshot(self):
        return dict(
            merged_adapter=self.mode,
            merge_initialized=self.merge_initialized,
            transitioning=self.transitioning,
            pending=len(self.pending),
            running=len(self.running),
            capacity=self.profile.capacity,
            failed=self.failure,
            completed=dict(self.completed),
            transitions=list(self.events[-128:]),
            profile=vars(self.profile),
            transition_totals=dict(self.transitions),
            rejections=self.rejections,
        )

    async def submit(self, adapter, prompt_tokens, output_tokens, prepare, execute):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.active >= self.profile.capacity + self.profile.queue_buffer:
            self.rejections += 1
            raise ReplicaBusy
        item = Pending(
            adapter,
            prompt_tokens,
            output_tokens,
            time.monotonic(),
            prepare,
            execute,
            asyncio.get_running_loop().create_future(),
        )
        self.pending.append(item)
        self.changed.set()
        try:
            return await item.future
        finally:
            if item in self.pending:
                self.pending.remove(item)
            for task, entry in tuple(self.running.items()):
                if entry is item and item.future.cancelled():
                    task.cancel()
            self.changed.set()

    def merge_candidate(self):
        """Amortize a complete merge/unmerge cycle using only admitted work.

        Charge all candidate prompts as uncached after the transition, even if
        they currently match. This conservative guard avoids trading a warm
        long prefix for a small decode saving. No predicted future arrivals.
        """
        if not self.pending or not self.profile.enabled:
            return None
        counts = Counter(e.adapter for e in self.pending)
        adapter, count = counts.most_common(1)[0]
        if adapter == "base":
            return None
        p = self.profile
        now = time.monotonic()
        others = [e for e in self.pending if e.adapter != adapter]
        if any(now - e.arrived >= p.max_wait_seconds for e in others):
            return None
        batch = [e for e in self.pending if e.adapter == adapter][: p.capacity]
        mixed = self.pending[: p.capacity]
        # Compare the sum of completion times for two feasible local plans:
        # mixed unmerged batch, or a homogeneous merged batch then the rest.
        # Partial batches get proportionally less credit for merge savings.
        fill = min(1.0, len(batch) / p.capacity)
        merged_step = p.unmerged_token_seconds + fill * (
            p.merged_token_seconds - p.unmerged_token_seconds
        )
        merged_finish = max(e.output_tokens for e in batch) * merged_step
        merge_seconds = (
            p.merge_seconds
            if self.merge_initialized
            else (
                p.initial_merge_seconds
                if p.initial_merge_seconds is not None
                else p.merge_seconds
            )
        )
        merged_finish += (
            merge_seconds
            + sum(e.prompt_tokens for e in batch) / p.prefill_tokens_per_second
        )
        unmerged_finish = max(e.output_tokens for e in mixed) * p.unmerged_token_seconds
        baseline = unmerged_finish * len(mixed)
        minority = [e for e in mixed if e.adapter != adapter]
        specialized = merged_finish * len(mixed)
        if minority:
            specialized += (
                p.unmerge_seconds
                + max(e.output_tokens for e in minority) * p.unmerged_token_seconds
            ) * len(minority)
        else:
            # Pay the eventual restoration even when no minority has arrived.
            specialized += p.unmerge_seconds * len(mixed)
        return adapter if specialized < baseline else None

    async def _transition(self, adapter):
        assert not self.running
        self.transitioning = True
        before = self.mode
        started = time.monotonic()
        try:
            if self.mode is not None:
                response = await self.client.post("/unmerge_lora_weights", json={})
                response.raise_for_status()
                if not response.json()["success"]:
                    raise RuntimeError(response.text)
                self.mode = None
                self.invalidate()
            if adapter is not None:
                entry = next(e for e in self.pending if e.adapter == adapter)
                await entry.prepare()
                response = await self.client.post(
                    "/merge_lora_weights", json={"lora_name": adapter}
                )
                response.raise_for_status()
                if not response.json()["success"]:
                    raise RuntimeError(response.text)
                self.mode = adapter
                self.merge_initialized = True
                self.invalidate()
            self.events.append(
                dict(
                    time=time.time(),
                    before=before,
                    after=self.mode,
                    seconds=time.monotonic() - started,
                )
            )
            self.transitions["count"] += 1
            self.transitions["seconds"] += time.monotonic() - started
            self.events[:] = self.events[-128:]
        finally:
            self.transitioning = False

    async def _execute(self, entry):
        try:
            result = await entry.execute()
            if not entry.future.done():
                entry.future.set_result(result)
            self.completed["merged" if self.mode else "unmerged"] += 1
        except asyncio.CancelledError:
            entry.future.cancel()
            raise
        except Exception as exc:
            if not entry.future.done():
                entry.future.set_exception(exc)
        finally:
            self.running.pop(asyncio.current_task(), None)
            self.changed.set()

    async def _run(self):
        try:
            while True:
                await self.changed.wait()
                self.changed.clear()
                if not self.running:
                    # Gather concurrent arrivals without a cluster round trip.
                    await asyncio.sleep(
                        min(0.02, self.profile.unmerged_token_seconds)
                        if self.profile.enabled
                        else 0.002
                    )
                self.pending[:] = [e for e in self.pending if not e.future.done()]
                if not self.pending:
                    continue
                if self.mode and any(e.adapter != self.mode for e in self.pending):
                    if self.running:
                        # Drain, rather than indefinitely refilling the hot adapter.
                        continue
                    await self._transition(None)
                if not self.running and self.mode is None:
                    candidate = self.merge_candidate()
                    if candidate:
                        await self._transition(candidate)
                room = self.profile.capacity - len(self.running)
                # Oldest first is the fairness credit: incompatible demand stops
                # merged refill above. No adapter can be skipped indefinitely.
                for entry in list(self.pending):
                    if room == 0:
                        break
                    if self.mode is not None and entry.adapter != self.mode:
                        continue
                    self.pending.remove(entry)
                    task = asyncio.create_task(self._execute(entry))
                    self.running[task] = entry
                    room -= 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # An uncertain transition must never fall through to generation.
            self.failure = f"Replica mode controller failed: {exc}"
            for entry in self.pending:
                if not entry.future.done():
                    entry.future.set_exception(RuntimeError(self.failure))
            self.pending.clear()

    async def aclose(self):
        self.worker.cancel()
        tasks = list(self.running)
        for task in tasks:
            task.cancel()
        for entry in self.pending:
            entry.future.cancel()
        await asyncio.gather(self.worker, *tasks, return_exceptions=True)
