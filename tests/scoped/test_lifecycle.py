import asyncio
import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from spindle.engines import qwen3_5_4b_full_64k, qwen3_6_27b_full_64k
from spindle.run import Pool


def test_pool_bounds():
    with pytest.raises(ValueError):
        Pool(min_containers=2, max_containers=1)


@pytest.mark.parametrize(
    "warm,body_failure,drain_failure,startup_failure,parent_failure,"
    "tag_failure,expected",
    [
        (True, None, False, False, None, None, "completed"),
        (False, "error", False, False, None, None, "failed"),
        (False, None, False, True, None, None, "failed"),
        (True, None, True, False, None, None, "failed"),
        (False, "interrupt", False, False, None, None, "cancelled"),
        (False, "cancel", False, False, None, None, "cancelled"),
        (True, None, False, False, None, "read", "completed"),
        (True, None, False, False, None, "write", "completed"),
        (False, "error", False, False, None, "read", "failed"),
        (False, "error", False, False, None, "write", "failed"),
        (True, None, False, False, "error", None, "failed"),
        (True, None, False, False, "interrupt", None, "cancelled"),
        (True, None, False, False, "cancel", None, "cancelled"),
        (True, None, False, False, "error", "write", "failed"),
    ],
)
def test_owned_children_stop_before_parent(
    monkeypatch,
    warm,
    body_failure,
    drain_failure,
    startup_failure,
    parent_failure,
    tag_failure,
    expected,
):
    import modal

    module = importlib.import_module("spindle.run")
    scoped = importlib.import_module("spindle.providers.modal.scoped")
    events = []
    data = {}
    base_tags = {
        "_modal_source": "spindle",
        "spindle_commit": "abc123",
        "spindle_role": "scoped",
        "_modal_model_name": "qwen3.5-4b",
        "team": "train",
    }
    written = {}

    class Registry:
        def put(self, key, value):
            data[key] = value

        def get(self, key):
            return data.get(key)

    def delete_registry(*args, **kwargs):
        events.append("registry-delete")

    monkeypatch.setattr(modal.Dict, "from_name", lambda *a, **k: Registry())
    monkeypatch.setattr(modal.Dict, "objects", SimpleNamespace(delete=delete_registry))
    monkeypatch.setattr(modal.Secret, "from_name", lambda *a, **k: None)

    parent_exc = {
        "error": RuntimeError("parent shutdown failure"),
        "interrupt": KeyboardInterrupt("parent interrupt"),
        "cancel": asyncio.CancelledError("parent cancel"),
    }.get(parent_failure)

    @contextmanager
    def parent():
        events.append("parent-start")
        try:
            yield
        except KeyboardInterrupt:
            pass
        finally:
            events.append("parent-stop")
            if parent_exc is not None:
                raise parent_exc

    def get_tags():
        events.append("read-tags")
        if tag_failure == "read":
            raise RuntimeError("tag read failure")
        if "parent-stop" in events:
            return {}
        return dict(base_tags)

    def set_tags(tags):
        events.append("write-tags")
        if tag_failure == "write":
            raise RuntimeError("tag write failure")
        written.update(tags)

    class Owner:
        def __init__(self, *args):
            pass

        def close(self):
            events.append("stop-pinned")
            if drain_failure:
                raise RuntimeError("drain failure")

    monkeypatch.setattr("spindle.providers.modal.scoped_pin_owner.PinOwner", Owner)

    def manage(action):
        events.append(action)
        if action == "close" and drain_failure:
            raise RuntimeError("drain failure")

    manage.aio = manage

    def prepare():
        events.append("assets")
        if startup_failure:
            raise RuntimeError("prepare failure")

    monkeypatch.setattr(
        scoped,
        "build_app",
        lambda *a, **kw: (
            SimpleNamespace(run=parent, get_tags=get_tags, set_tags=set_tags),
            SimpleNamespace(get_web_url=lambda: "https://example.invalid"),
            SimpleNamespace(remote=manage),
            [],
            SimpleNamespace(remote=prepare),
            SimpleNamespace(object_id="im-test"),
        ),
    )
    body_exc = {
        "error": ValueError("body failure"),
        "interrupt": KeyboardInterrupt(),
        "cancel": asyncio.CancelledError(),
    }.get(body_failure)
    caught = None
    try:
        with module.run(engine=qwen3_5_4b_full_64k(), warm=warm) as (url, key):
            assert url == "https://example.invalid" and key.startswith("tml-")
            if body_exc is not None:
                raise body_exc
    except (
        ValueError,
        RuntimeError,
        ExceptionGroup,
        KeyboardInterrupt,
        asyncio.CancelledError,
    ) as exc:
        caught = exc
    assert (caught is not None) == bool(
        body_failure or drain_failure or startup_failure or parent_failure
    )
    if body_exc is not None:
        assert caught is body_exc
    if parent_exc is not None:
        assert caught is parent_exc
    assert ("warm" in events) == warm
    assert events.index("stop-pinned") < events.index("parent-stop")
    if tag_failure:
        assert not written
    else:
        assert written == {**base_tags, "spindle_scope_status": expected}
    if "read-tags" in events:
        assert events.index("read-tags") < events.index("parent-stop")
    if "write-tags" in events:
        assert events.index("write-tags") > events.index("parent-stop")
    if "registry-delete" in events and "write-tags" in events:
        assert events.index("write-tags") > events.index("registry-delete")


def test_recipes_keep_context_and_topology_together():
    small = qwen3_5_4b_full_64k()
    large = qwen3_6_27b_full_64k()
    small.validate()
    large.validate()
    assert small.training.fp32_lm_head
    assert small.training.optimizer.loss_scale == 1
    assert large.training.tensor_model_parallel_size == 4
    assert large.training.context_parallel_size == 2
    assert large.training.seq_length == 65536
    assert large.training.provider_overrides["recompute_granularity"] == "full"


def test_publication_targets_the_scoped_latest_route(monkeypatch):
    import modal

    from spindle.providers.modal.scoped_pool import ScopedFlashPool, publication_pool

    route = {"url": "https://sampler.invalid", "function_id": "fu-scoped"}
    monkeypatch.setenv("SPINDLE_SCOPED_REGISTRY", "owned-run")
    monkeypatch.setattr(modal.Dict, "from_name", lambda name: {"model:abc": route})
    pool = publication_pool("custom-engine", "abc")
    assert isinstance(pool, ScopedFlashPool)
    assert pool.route == route
    assert pool.gateway_url() == route["url"]


def test_shared_publication_keeps_existing_pool(monkeypatch):
    from spindle.providers.modal.fft_pool import FFTLatestPool
    from spindle.providers.modal.scoped_pool import publication_pool

    monkeypatch.delenv("SPINDLE_SCOPED_REGISTRY", raising=False)
    assert isinstance(publication_pool("existing", "abc"), FFTLatestPool)


def test_latest_minimum_updates_by_id_without_name_lookup(monkeypatch):
    from modal.client import _Client

    from spindle.providers.modal.scoped_pool import set_minimum

    calls = []

    async def update(request):
        calls.append(request)

    async def client():
        return SimpleNamespace(
            stub=SimpleNamespace(FunctionUpdateSchedulingParams=update)
        )

    monkeypatch.setattr(_Client, "from_env", client)
    set_minimum("fu-ephemeral", 1)
    assert calls[0].function_id == "fu-ephemeral"
    assert calls[0].settings.min_containers == 1


def test_retry_finishes_preparation_without_creating_another_model():
    import asyncio

    from spindle.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
    from spindle.providers.modal.scoped_control import ScopedControlPlane
    from tests.support import EchoExecutor

    prepared = []

    async def prepare(model):
        prepared.append(model.model_id)
        if len(prepared) == 1:
            raise RuntimeError("transient infrastructure error")

    async def check():
        plane = ScopedControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform("test", EchoExecutor),
            prepare_model=prepare,
        )
        session = await plane.create_session()
        args = dict(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id="test",
            spec={"rank": 32},
        )
        with pytest.raises(RuntimeError):
            await plane.create_model(**args)
        result = await plane.create_model(**args)
        assert prepared == [result.model.model_id] * 2
        assert not result.created

    asyncio.run(check())
