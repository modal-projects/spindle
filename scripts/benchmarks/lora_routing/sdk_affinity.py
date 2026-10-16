"""Add Spindle's affinity extension while retaining the normal Tinker SDK path."""

import hashlib
import json

import httpx


def with_prompt_affinity(request):
    if request.url.path != "/api/v1/asample":
        return request
    body = json.loads(request.content)
    if "cache_affinity_key" not in body:
        encoded = json.dumps(body["prompt"], sort_keys=True, separators=(",", ":"))
        body["cache_affinity_key"] = hashlib.sha256(encoded.encode()).hexdigest()
    headers = dict(request.headers)
    headers.pop("content-length", None)
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        json=body,
        extensions=request.extensions,
    )


class PromptAffinityTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request):
        await request.aread()
        return await super().handle_async_request(with_prompt_affinity(request))
