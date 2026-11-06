"""Read the serving publication identity; Miles uses command sequence IDs."""

import asyncio

import modal
from stitch.types import VersionRef


async def latest_ref(model_id, volume_name):
    volume = modal.Volume.from_name(volume_name, environment_name="kailash-dev")
    pointer = await asyncio.to_thread(
        lambda: b"".join(volume.read_file(model_id + "/latest"))
    )
    return VersionRef.parse(pointer.decode().strip())
