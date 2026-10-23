"""Discover Modal workers; the native router owns observations and dispatch."""

from stitch.pools.modal_flash import list_flash_containers_async

from spindle.inference.native_router import NativeRouter

from .fft_pool import proxy_auth_headers


def create_router(app_name, gateway, **settings):
    async def discover_addresses():
        containers = await list_flash_containers_async(app_name, "Server")
        return [
            dict(
                replica=c.task_id,
                upstream=f"{c.host}:{c.port}" if c.port else c.host,
            )
            for c in containers
            if c.host
        ]

    return NativeRouter(
        discover_addresses,
        gateway=gateway,
        headers=proxy_auth_headers(),
        **settings,
    )
