import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from spindle.providers.modal.kv import ModalVolumeStore


class FakeVolume:
    """The slice of modal.Volume the registry uses, over a dict of files."""

    def __init__(self):
        self.files = {}
        self.listdir = SimpleNamespace(aio=self._listdir)
        self.remove_file = SimpleNamespace(aio=self._remove)
        self.batch_upload = SimpleNamespace(aio=self._batch)

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    async def _listdir(self, path):
        return [SimpleNamespace(path=name) for name in self.files]

    async def _remove(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    @asynccontextmanager
    async def _batch(self, force=False):
        volume = self

        class Batch:
            def put_file(self, fileobj, remote_path):
                volume.files[remote_path.lstrip("/")] = fileobj.read()

        yield Batch()


def test_volume_store_keeps_one_json_file_per_recipe():
    async def run():
        volume = FakeVolume()
        store = ModalVolumeStore(volume)
        assert await store.get("missing") is None
        await store.put("qwen-lora", {"config": {"recipe": {"name": "qwen-lora"}}})
        await store.put("qwen-fft", {"config": {"recipe": {"name": "qwen-fft"}}})
        assert set(volume.files) == {"qwen-lora.json", "qwen-fft.json"}
        assert volume.files["qwen-lora.json"].startswith(b"{")
        assert store.read("qwen-lora") == {"config": {"recipe": {"name": "qwen-lora"}}}
        assert await store.list_keys("qwen") == ("qwen-fft", "qwen-lora")
        assert [key for key, _ in await store.list_items("")] == [
            "qwen-fft",
            "qwen-lora",
        ]
        await store.delete("qwen-fft")
        await store.delete("qwen-fft")
        assert await store.list_keys("") == ("qwen-lora",)

    asyncio.run(run())
