from array import array
import json

import pytest
import torch

from test_routing_observer import (
    await_ready,
    expected,
    fake_scheduler,
    native_cache,
    observer,  # noqa: F401 (pytest fixture)
    optimized,
)

native = pytest.importorskip("sglang.srt.mem_cache.rust_tree_core.extension").bindings


def core(hybrid=False, bigram=False):
    cls = (
        native.RustBigramUnifiedTreeCoreBinding
        if bigram
        else native.RustUnifiedTreeCoreBinding
    )
    return cls(
        native.TreeCoreInitParamsBinding(mamba_cache_chunk_size=64 if hybrid else None),
        [0, 2] if hybrid else [0],
    )


def insert(tree, tokens, adapter="a", hybrid=False, salt=None):
    return tree.insert(
        native.InsertParamsBinding(
            array("q", tokens),
            torch.arange(len(tokens), dtype=torch.int64),
            extra_key=adapter,
            cache_salt=salt,
            mamba_value=torch.tensor([1], dtype=torch.int64) if hybrid else None,
        )
    )


def full_export(tree, names, hybrid):
    root, rows = tree.routing_snapshot_delta({})
    unpacked = []
    for node, parent, adapter, packed, mamba, accessed, _ in rows:
        tokens = array("q")
        tokens.frombytes(packed)
        unpacked.append((node, parent, adapter, tokens.tolist(), mamba, accessed))
    return expected(unpacked, names, hybrid, root)


@pytest.mark.parametrize("hybrid", [False, True])
def test_native_mutations_match_full_export(hybrid):
    tree = core(hybrid)
    consumer = optimized.IncrementalPrefixes()
    names = {"a": "adapter-a-v1", "b": "adapter-b-v1"}
    known = {}

    def check():
        nonlocal known
        root, rows = tree.routing_snapshot_delta(known)
        actual = consumer.export(root, rows, names, hybrid)
        assert actual == full_export(tree, names, hybrid)
        known = {r[0]: r[6] for r in rows}
        # Snapshotting must not modify access counters or any tree data.
        before = tree.routing_snapshot_delta({})
        _, unchanged = tree.routing_snapshot_delta(known)
        assert all(row[3] is None for row in unchanged)
        assert tree.routing_snapshot_delta({}) == before
        return rows

    insert(tree, list(range(256)), hybrid=hybrid)
    check()
    insert(tree, list(range(96)) + [1000] * 160, hybrid=hybrid)
    assert any(row[3] is not None for row in check())
    insert(tree, list(range(256)), adapter="b", hybrid=hybrid)
    check()
    names["a"] = "adapter-a-v2"
    check()
    leaf = insert(tree, [9000] * 128, hybrid=hybrid).last_device_node
    check()
    tree.drop_subtree_no_host(leaf)
    check()
    tree.reset()
    check()
    insert(tree, [8000] * 128, hybrid=hybrid)
    check()


def test_salted_bigram_host_only_excluded():
    tree = core()
    insert(tree, [1] * 128, salt="private")
    assert tree.routing_snapshot_delta({})[1] == []
    tree.insert_host(
        tree.root_node_handle(None),
        "a",
        array("q", [2] * 128),
        torch.arange(128, dtype=torch.int64),
        [],
    )
    assert tree.routing_snapshot_delta({})[1] == []
    bigram = core(bigram=True)
    insert(bigram, [1] * 129)
    assert bigram.routing_snapshot_delta({})[1] == []


@pytest.mark.usefixtures("observer")
def test_scheduler_integration(tmp_path):
    tree = core(hybrid=True)
    insert(tree, list(range(256)), hybrid=True)
    # Use actual native state behind the scheduler-facing cache interface.
    cache = native_cache(tree)
    cache.is_mamba_enabled = True
    scheduler = fake_scheduler(cache)
    await_ready(optimized.publisher, cache)
    optimized.snapshot(scheduler, None, None)
    await_ready(optimized.publisher, cache)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["prefixes"] == full_export(tree, {"a": "adapter-a-v1"}, True)
    assert state["kv_capacity_tokens"] == 1512
    assert state["export_token_bytes"] == 256 * 8
    optimized.last_snapshot = 0
    optimized.snapshot(scheduler, None, None)
    await_ready(optimized.publisher, cache)
    assert json.loads((tmp_path / "state.json").read_text())["export_token_bytes"] == 0
