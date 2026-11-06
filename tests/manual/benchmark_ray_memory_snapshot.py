"""CPU-only control for the 9B trainer snapshot experiment."""

import contextlib
import json
import os
import time
import uuid
from pathlib import Path

import modal

APP_NAME = "spindle-ray-memory-snapshot-control-1105"
app = modal.App(APP_NAME)
if modal.is_local():
    from spindle.providers.modal.miles_image import image
else:
    image = modal.Image.debian_slim()


@app.cls(
    image=image,
    cpu=4,
    memory=16384,
    enable_memory_snapshot=True,
    single_use_containers=True,
    max_containers=1,
    startup_timeout=300,
    timeout=120,
    retries=0,
)
class RayControl:
    node_ip: str = modal.parameter(default="auto")

    @modal.enter(snap=True)
    def initialize(self):
        import ray

        self.initialization_id = uuid.uuid4().hex
        options = {} if self.node_ip == "auto" else {"_node_ip_address": self.node_ip}
        if self.node_ip == "127.0.0.1":
            # Ray otherwise rewrites localhost to the container's external IP.
            ray._private.services.resolve_ip_for_localhost = lambda host: host
        ray.init(num_cpus=4, include_dashboard=False, **options)
        self.bound_ip = ray.nodes()[0]["NodeManagerAddress"]
        if self.node_ip != "auto":
            assert self.bound_ip == self.node_ip, self.bound_ip

        @ray.remote
        class Counter:
            def __init__(self):
                self.value = 17

            def read(self):
                return os.getpid(), self.value

            def increment(self):
                self.value += 1
                return self.value

        self.actors = [Counter.remote() for _ in range(4)]
        self.before = ray.get([actor.read.remote() for actor in self.actors])
        self.ray_version = ray.__version__
        print("RAY_SNAPSHOT_READY", self.initialization_id, self.before, flush=True)

    @modal.enter(snap=False)
    def restored(self):
        self.boot_id = uuid.uuid4().hex
        print("RAY_POST_RESTORE", self.boot_id, flush=True)

    @modal.method()
    def verify(self):
        import ray

        actual = ray.get([actor.read.remote() for actor in self.actors], timeout=60)
        assert actual == self.before, (actual, self.before)
        assert ray.get([actor.increment.remote() for actor in self.actors]) == [18] * 4
        return {
            "initialization_id": self.initialization_id,
            "boot_id": self.boot_id,
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "ray_version": self.ray_version,
            "node_ip": self.node_ip,
            "bound_ip": self.bound_ip,
            "actors": actual,
        }

    @modal.exit()
    def close(self):
        import ray

        ray.shutdown()


@app.local_entrypoint()
def main(
    node_ip: str = "auto",
    repeats: int = 2,
    output: str = "/tmp/ray-memory-snapshot.json",
):
    cls = modal.Cls.from_name(APP_NAME, "RayControl")
    rows = []
    for _ in range(repeats):
        started = time.monotonic()
        call = cls(node_ip=node_ip).verify.spawn()
        print("RAY_SNAPSHOT_CALL", call.object_id, flush=True)
        try:
            result = call.get(timeout=600)
        except BaseException as exc:
            with contextlib.suppress(Exception):
                call.cancel(terminate_containers=True)
            Path(output).write_text(
                json.dumps(
                    {
                        "rows": rows,
                        "failure": {
                            "call_id": call.object_id,
                            "error": str(exc),
                            "elapsed_s": time.monotonic() - started,
                        },
                    },
                    indent=2,
                )
            )
            raise
        result["call_to_result_s"] = time.monotonic() - started
        rows.append(result)
        Path(output).write_text(json.dumps(rows, indent=2))
        print("RAY_SNAPSHOT_RESULT", json.dumps(result), flush=True)
