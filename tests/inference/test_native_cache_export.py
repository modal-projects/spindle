from array import array
from types import SimpleNamespace

import pytest
import torch

from test_routing_observer import optimized, expected, await_ready

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


def test_scheduler_integration(tmp_path):
    tree = core(hybrid=True)
    insert(tree, list(range(256)), hybrid=True)
    # Use actual native state behind the scheduler-facing cache interface.
    adapter = type("RustUnifiedTreeCore", (), {})()
    adapter._binding = tree
    cache = type("UnifiedRadixCache", (), {})()
    cache.tree_core = adapter
    cache.is_swa_enabled = False
    cache.is_mamba_enabled = True
    manager = SimpleNamespace(
        lora_refs={"a": SimpleNamespace(lora_name="adapter-a-v1")},
        loras={"a": None},
        memory_pool=SimpleNamespace(uid_to_buffer_id={"a": 0}),
    )
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(tp_rank=0, dp_size=1, pp_size=1),
        enable_lora=True,
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(lora_manager=manager)),
        tree_cache=cache,
        waiting_queue=[],
        chunked_req=None,
        pool_stats_observer=SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                full_num_used=256, full_available_size=1000, full_evictable_size=256
            )
        ),
    )
    import json

    publisher = optimized.BackgroundPublisher(tmp_path / "state.json")
    optimized.publisher = publisher
    optimized.ENABLED = True
    try:
        await_ready(publisher, cache)
        optimized.last_snapshot = 0
        optimized.snapshot(scheduler, None, None)
        await_ready(publisher, cache)
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["prefixes"] == full_export(tree, {"a": "adapter-a-v1"}, True)
        assert state["kv_capacity_tokens"] == 1512
        assert state["export_token_bytes"] == 256 * 8
        optimized.last_snapshot = 0
        optimized.snapshot(scheduler, None, None)
        await_ready(publisher, cache)
        assert (
            json.loads((tmp_path / "state.json").read_text())["export_token_bytes"] == 0
        )
    finally:
        publisher.close()
