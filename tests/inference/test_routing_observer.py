from array import array
import json
import random
import signal
import time

import pytest
import xxhash

optimized = pytest.importorskip("sglang.srt.spindle_routing_observer")


def expected(rows, names, hybrid, root=0):
    # Independent oracle: rebuild each complete prefix, then hash it directly.
    prefixes = set()

    def visit(parent, prefix):
        for node, ancestor, adapter, tokens, mamba, _ in rows:
            if ancestor != parent or adapter not in names:
                continue
            whole = prefix + tokens
            for end in range((len(prefix) // 64 + 1) * 64, len(whole) + 1, 64):
                if not hybrid or (mamba and end == len(whole)):
                    digest = xxhash.xxh3_128((names[adapter] + "\0").encode())
                    digest.update(array("q", whole[:end]).tobytes())
                    prefixes.add(digest.hexdigest())
            visit(node, whole)

    visit(root, [])
    return sorted(prefixes)


def packed(rows, known=None):
    known = known or {}
    return [
        (
            node,
            parent,
            adapter,
            None if known.get(node) == len(tokens) else array("q", tokens).tobytes(),
            mamba,
            accessed,
            len(tokens),
        )
        for node, parent, adapter, tokens, mamba, accessed in rows
    ]


@pytest.mark.parametrize("hybrid", [False, True])
def test_random_trees_match_baseline_with_incremental_updates(hybrid):
    rng = random.Random(173)
    names = {"a": "adapter-a-v1", "b": "adapter-b-v1"}
    state = optimized.IncrementalPrefixes()
    rows = []
    known = {}
    for step in range(60):
        parent = rng.choice([0] + [row[0] for row in rows])
        adapter = (
            rng.choice(["a", "b", "unknown"])
            if parent == 0
            else next(r[2] for r in rows if r[0] == parent)
        )
        rows.append(
            (
                step + 1,
                parent,
                adapter,
                [rng.randrange(100000) for _ in range(rng.randrange(1, 513))],
                bool(step % 3),
                step,
            )
        )
        delta = packed(rows, known)
        assert state.export(0, delta, names, hybrid) == expected(rows, names, hybrid)
        known = {row[0]: len(row[3]) for row in rows}
    # Adapter version changes must invalidate prefix hashes even with no new tokens.
    names["a"] = "adapter-a-v2"
    assert state.export(0, packed(rows, known), names, hybrid) == expected(
        rows, names, hybrid
    )
    # Removing a live ancestor suppresses all its descendants' hints.
    rows = rows[1:]
    assert state.export(0, packed(rows, known), names, hybrid) == expected(
        rows, names, hybrid
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_split_eviction_and_reset(hybrid):
    names = {"a": "adapter-v1"}
    rows = [
        (1, 0, "a", list(range(256)), True, 1),
        (2, 1, "a", list(range(256, 512)), True, 2),
    ]
    state = optimized.IncrementalPrefixes()
    assert state.export(0, packed(rows), names, hybrid) == expected(rows, names, hybrid)
    known = {1: 256, 2: 256}
    rows = [
        (3, 0, "a", list(range(96)), False, 3),
        (1, 3, "a", list(range(96, 256)), True, 4),
        rows[1],
    ]
    assert state.export(0, packed(rows, known), names, hybrid) == expected(
        rows, names, hybrid
    )
    # Parent changes while descendant length stays unchanged.
    rows = rows[:2]
    assert state.export(
        0, packed(rows, {3: 96, 1: 160, 2: 256}), names, hybrid
    ) == expected(rows, names, hybrid)
    assert set(state.tokens) == {1, 3}
    rows = [(8, 7, "a", [42] * 128, True, 1)]
    assert state.export(7, packed(rows), names, hybrid) == expected(
        rows, names, hybrid, root=7
    )
    assert set(state.tokens) == {8}


def test_catalog_keeps_short_ancestors_when_recent_endpoints_fill_it():
    # The two most recent endpoints omit 'a'; the ancestry budget retains it.
    boundaries = {"a": (None, True, 1), "b": ("a", True, 3), "c": ("b", True, 4)}
    assert optimized.select_prefixes(boundaries, limit=2) == ["a", "b", "c"]
    boundaries["a"] = (None, False, 1)
    assert optimized.select_prefixes(boundaries, limit=2) == ["b", "c"]


def test_missing_delta_is_rejected():
    with pytest.raises(ValueError, match="Incomplete routing delta"):
        optimized.IncrementalPrefixes().export(
            0, [(1, 0, "a", None, True, 1, 64)], {"a": "adapter"}, True
        )


def await_ready(publisher, tree):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if publisher.ready(tree):
            return
        time.sleep(0.01)
    raise AssertionError("publisher did not become ready")


def packet(sequence, rows, timestamp=None):
    return (
        {
            "sequence": sequence,
            "timestamp": time.time() if timestamp is None else timestamp,
        },
        0,
        rows,
        {"a": "adapter"},
        True,
    )


def test_publisher_stale_snapshot_restart_and_inventory(tmp_path):
    path = tmp_path / "observations.json"
    publisher = optimized.BackgroundPublisher(path)
    tree = object()
    rows = [(1, 0, "a", list(range(128)), True, 1)]
    try:
        await_ready(publisher, tree)
        publisher.submit(packet(1, packed(rows)))
        assert not publisher.ready(tree)
        await_ready(publisher, tree)
        first = json.loads(path.read_text())
        assert first["sequence"] == 1
        assert publisher.known == {1: 128}
        publisher.submit(
            packet(2, packed(rows, publisher.known), timestamp=time.time() - 10)
        )
        await_ready(publisher, tree)
        assert json.loads(path.read_text())["sequence"] == 1
        publisher.process.kill()
        publisher.process.wait()
        publisher.submit(packet(3, packed(rows, publisher.known)))
        await_ready(publisher, tree)
        assert publisher.known == {}
        assert publisher.failures == 1
        publisher.submit(packet(4, packed(rows)))
        await_ready(publisher, tree)
        assert json.loads(path.read_text())["sequence"] == 4
        assert json.loads(path.read_text())["prefixes"] == first["prefixes"]
        assert publisher.ready(object())
        assert publisher.known == {}
    finally:
        publisher.close()
        publisher.thread.join(timeout=3)
    assert not publisher.thread.is_alive()


def test_stalled_publisher_times_out_without_blocking_caller(tmp_path):
    publisher = optimized.BackgroundPublisher(tmp_path / "state.json")
    tree = object()
    try:
        await_ready(publisher, tree)
        publisher.process.send_signal(signal.SIGSTOP)
        # Larger than the socket buffer, to test a blocked send as well as a
        # missing acknowledgement. The scheduler only enqueues the immutable data.
        rows = [(1, 0, "a", b"\0" * 2_000_000, True, 1, 250000)]
        started = time.monotonic()
        publisher.submit(packet(1, rows))
        assert time.monotonic() - started < 0.1
        assert not publisher.ready(tree)
        await_ready(publisher, tree)
        assert publisher.failures == 1
        assert publisher.known == {}
    finally:
        publisher.close()
        publisher.thread.join(timeout=3)
    assert not publisher.thread.is_alive()
