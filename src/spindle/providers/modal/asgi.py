"""Run an ASGI application for the lifetime of a Modal Server container."""

from __future__ import annotations

import threading
import time

import uvicorn


class ASGIServer:
    def __init__(self, app, *, port=8000):
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                limit_concurrency=128,
                timeout_graceful_shutdown=25,
            )
        )
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        try:
            self.server.run()
        except BaseException as exc:
            self.error = exc

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 45
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("API server exited during startup") from self.error
            if time.monotonic() >= deadline:
                self.stop()
                raise TimeoutError("API server did not start within 45 seconds")
            time.sleep(0.01)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=28)
        if self.thread.is_alive():
            raise RuntimeError("API server did not stop within 28 seconds")
