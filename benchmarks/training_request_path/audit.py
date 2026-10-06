"""Benchmark-only backend: audit received data, then execute real GPU training."""

import hashlib
import json
import time

from spindle.backends.miles_lora import build_executor as build_miles_executor
from spindle.engine.operations import serialize_operation_payload
from spindle.engine.spmd import DistributedExecutor
from spindle.request_timing import mark


class AuditedExecutor(DistributedExecutor):
    async def execute_forward_backward_batch(self, executions):
        arrived = time.time()
        # Kept identical in both arms and timed separately. This intentionally
        # uses the old JSON representation as an independent equality oracle.
        for command in executions:
            payload = serialize_operation_payload(command.payload)
            digest = hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            mark(
                "benchmark.backend_payload",
                model_id=command.model_id,
                sha256=digest,
                examples=len(command.payload.data),
                arrived=arrived,
            )
        mark("benchmark.audit_done", seconds=time.time() - arrived)
        return await super().execute_forward_backward_batch(executions)


def build_executor():
    original = build_miles_executor()
    return AuditedExecutor(original.backend)
