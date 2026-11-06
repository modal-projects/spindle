"""Diagnostic hooks around the production backend executor, isolated to the probe."""

import asyncio
import os
import sys
import time
import uuid


def build_executor():
    from spindle.backends import miles_runtime
    from spindle.backends.miles_lora import build_executor as production_build_executor

    runtime = miles_runtime.runtime
    runtime.ray._private.services.resolve_ip_for_localhost = lambda host: host
    original_init = runtime.ray.init

    def local_init(*args, **kwargs):
        kwargs.setdefault("_node_ip_address", "127.0.0.1")
        return original_init(*args, **kwargs)

    runtime.ray.init = local_init
    runtime._configure_actor_spec(runtime.train_specs)
    original_spec = runtime.train_specs._compute_spec_trainer

    def actor_spec(*args, **kwargs):
        spec = original_spec(*args, **kwargs)
        if spec.worker_class.endswith(".SpindleMilesTrainRayActor"):
            spec = spec.model_copy(
                update={"worker_class": "qwen9b_snapshot_probe.SnapshotProbeActor"}
            )
        return spec

    runtime.train_specs._compute_spec_trainer = actor_spec
    executor = production_build_executor()
    metadata = {
        "initialization_id": uuid.uuid4().hex,
        "creator_task_id": os.environ["MODAL_TASK_ID"],
        "backend_pid": os.getpid(),
        "executor_type": type(executor).__qualname__,
        "backend_type": type(executor.backend).__qualname__,
        "miles_revision": executor.backend.runtime.revision,
    }
    # backend_http runs as __main__. Extend its app with diagnostics only;
    # all training endpoints and the executor factory remain production code.
    main = sys.modules["__main__"]
    original_app = main.create_backend_app

    def create_app(*args, **kwargs):
        app = original_app(*args, **kwargs)

        @app.post("/snapshot_control/{command}")
        async def control(command: str):
            if command == "metadata":
                return {**metadata, "jobs": list(executor.backend.jobs)}
            methods = {
                "probe": "snapshot_probe",
                "suspend": "snapshot_suspend_comms",
                "resume": "snapshot_resume_comms",
            }
            method = methods[command]
            started = time.monotonic()

            def call():
                owner = executor.backend.runtime
                return owner._run(owner._trainer._execute_slots(method))

            result = await asyncio.to_thread(call)
            return {"result": result, "elapsed_s": time.monotonic() - started}

        return app

    main.create_backend_app = create_app
    return executor
