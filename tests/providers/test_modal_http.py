import asyncio
import inspect
import json
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException
from tinker import ServiceClient, TinkerError
from tinker.lib._auth_token_provider import ApiKeyAuthProvider

from scripts.check_modal_http import violations
from spindle.client import proxy_auth_headers, proxy_client_kwargs
from spindle.providers.modal.asgi import ASGIServer


@pytest.mark.parametrize(
    "source",
    [
        "@modal.asgi_app(requires_proxy_auth=True)\ndef api(): pass",
        "@m.web_server(8000)\ndef api(): pass",
        "from modal import fastapi_endpoint as endpoint\n@endpoint()\ndef api(): pass",
        "@modal.wsgi_app()\ndef api(): pass",
        "@modal.web_endpoint()\ndef api(): pass",
        "@app.server()\nclass Api: pass",
        "@app.server(unauthenticated=True)\nclass Api: pass",
        "@app.server(unauthenticated=0)\nclass Api: pass",
        "@app.server(unauthenticated=allow_public)\nclass Api: pass",
        "@app.server(**options)\nclass Api: pass",
        "def factory(app):\n @app.server(unauthenticated=True)\n class Api: pass",
    ],
)
def test_http_audit_rejects_web_functions_and_unsafe_servers(source):
    assert violations(source, "example.py")


def test_http_audit_accepts_explicit_authentication():
    assert not violations(
        "@app.server(unauthenticated=False, **placement)\nclass Api: pass",
        "example.py",
    )


def test_proxy_credentials_reach_supported_tinker_client(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    headers = {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}
    assert proxy_auth_headers() == headers
    options = proxy_client_kwargs()
    if "default_headers_overrides" in inspect.signature(ServiceClient).parameters:
        assert options == {"default_headers_overrides": headers}
    else:
        assert options == {"default_headers": headers}
    # Both versions initialize lazily; constructing the client requires no network.
    ServiceClient(base_url="http://unused.invalid", api_key="tml-test", **options)


def test_proxy_credentials_fail_closed_when_missing(monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    with pytest.raises(KeyError):
        proxy_client_kwargs()


def test_sdk_rejects_proxy_token_as_tinker_api_key():
    with pytest.raises(TinkerError, match="api_key must start with the 'tml-' prefix"):
        ApiKeyAuthProvider("wk-test.ws-test")


@pytest.mark.usefixtures("sdk_garbage_collected_off_loop")
def test_sdk_sends_tinker_key_and_proxy_headers_together(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(dict(self.headers))
            body = json.dumps({"supported_models": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service = ServiceClient(
        base_url=f"http://127.0.0.1:{server.server_port}",
        api_key="tml-test",
        **proxy_client_kwargs(),
    )
    try:
        assert service.get_server_capabilities().supported_models == []
        assert requests
        for headers in requests:
            headers = httpx.Headers(headers)
            assert headers["X-API-Key"] == "tml-test"
            assert headers["Modal-Key"] == "wk-test"
            assert headers["Modal-Secret"] == "ws-test"
    finally:
        if service._rest_holder:
            service._rest_holder.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_asgi_server_serves_requests_and_runs_lifespan():
    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("start")
        yield
        events.append("stop")

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def protected(x_api_key: str | None = Header(default=None)):
        if x_api_key != "tml-test":
            raise HTTPException(401)
        return {"status": "ok"}

    server = ASGIServer(app, port=0)
    server.start()
    try:
        port = server.server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            assert client.get("/").status_code == 401
            assert client.get("/", headers={"X-API-Key": "wrong"}).status_code == 401
            assert client.get("/", headers={"X-API-Key": "tml-test"}).json() == {
                "status": "ok"
            }
    finally:
        server.stop()
    assert events == ["start", "stop"]
    assert not server.thread.is_alive()


def test_asgi_server_reports_startup_failure():
    @asynccontextmanager
    async def lifespan(app):
        raise ValueError("bad startup")
        yield

    server = ASGIServer(FastAPI(lifespan=lifespan), port=0)
    with pytest.raises(RuntimeError, match="exited during startup"):
        server.start()
    assert not server.thread.is_alive()


def test_control_plane_key_required_for_api_routes():
    from spindle.control_plane.http import create_control_plane_app

    app = create_control_plane_app(object(), (), api_key="tml-test")

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for key in (None, "wrong", "tml-test"):
                response = await client.get(
                    "/api/v1/healthz", headers={"X-API-Key": key} if key else {}
                )
                assert response.status_code == (200 if key == "tml-test" else 401)

    asyncio.run(check())
