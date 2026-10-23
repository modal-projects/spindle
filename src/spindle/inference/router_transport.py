"""Sampling interface for a router that owns admission and dispatch.

Dispatch owns admission and reservation lifetime through response receipt.
Logical version validation remains with the sampling API. Implementations must
release reservations on cancellation and respect the caller's absolute deadline.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass
class RoutedResponse:
    response: httpx.Response
    replica: str | None
    routing: dict[str, Any]


class RouterTransport(Protocol):
    policy: str

    def prepare_request(
        self, adapter: str, affinity: str, prompt: list[int], output_tokens: int
    ) -> object: ...

    async def dispatch(
        self,
        prepared: object,
        *,
        request_id: str,
        gateway: str,
        body: dict,
        headers: dict,
        client: httpx.AsyncClient,
        deadline: float,
        trace_attrs: dict | None = None,
    ) -> RoutedResponse: ...

    async def aclose(self) -> None: ...
