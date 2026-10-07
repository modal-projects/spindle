"""Reproduce caller-poll starvation and check durable Modal call survival on CPU.

modal run tests/manual/validate_glm53_call_lifetime.py
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import modal
from modal._serialization import serialize, deserialize

app = modal.App("glm53-call-lifetime-validation")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    timeout=600,
    cpu=0.25,
    memory=512,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
async def delayed(seconds: int):
    await asyncio.sleep(seconds)
    return {"container": os.environ["MODAL_TASK_ID"], "seconds": seconds}


@app.local_entrypoint()
def main():
    delayed.remote(0)
    handle = tempfile.NamedTemporaryFile(prefix="glm53-call-", delete=False)
    handle.write(serialize(delayed))
    handle.close()
    processes = []
    try:
        for mode in ("remote", "durable"):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        handle.name,
                        mode,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        # Both calls are running when the caller stops polling for >130 seconds.
        time.sleep(15)
        for process in processes:
            os.kill(process.pid, signal.SIGSTOP)
        time.sleep(150)
        for process in processes:
            os.kill(process.pid, signal.SIGCONT)
        outputs = [p.communicate(timeout=180)[0] for p in processes]
        print(json.dumps(dict(zip(("remote", "durable"), outputs)), indent=2))
        assert processes[0].returncode != 0 and "cancelled" in outputs[0]
        assert processes[1].returncode == 0 and '"seconds": 180' in outputs[1]
        print("CALL LIFETIME VALIDATION PASSED")
    finally:
        Path(handle.name).unlink(missing_ok=True)
        for process in processes:
            if process.poll() is None:
                os.kill(process.pid, signal.SIGCONT)
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    fn = deserialize(Path(sys.argv[2]).read_bytes(), modal.Client.from_env())
    if sys.argv[3] == "durable":
        result = fn.spawn(180).get()
    else:
        result = fn.remote(180)
    print(json.dumps(result), flush=True)
