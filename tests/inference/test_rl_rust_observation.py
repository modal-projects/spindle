from types import SimpleNamespace
from collections import Counter

from spindle.inference import sglang_routing_observer as observer
from spindle.inference.sglang_routing_observer import rust_prefixes
from array import array
import xxhash


def token_prefixes(adapter, tokens):
    h = xxhash.xxh3_128((adapter + "\0").encode())
    result = []
    for end in range(64, len(tokens), 64):
        h.update(array("q", tokens[end - 64 : end]).tobytes())
        result.append((h.hexdigest(), end))
    return result


def cache(rows):
    return SimpleNamespace(
        is_mamba_enabled=True,
        tree_core=SimpleNamespace(
            root_node_handle=lambda: 0,
            _binding=SimpleNamespace(routing_snapshot=lambda: rows),
        ),
    )


def test_rust_observation_requires_exact_adapter_and_mamba_residency():
    rows = [
        (1, 0, "uid", list(range(64)), False, 1),
        (2, 1, "uid", list(range(64, 128)), True, 2),
        (3, 0, "other-uid", list(range(64)), True, 3),
    ]
    prefixes = rust_prefixes(cache(rows), {"uid": "model/7"})
    expected = dict((n, h) for h, n in token_prefixes("model/7", list(range(129))))
    assert prefixes == [expected[128]]
    assert not set(prefixes) & {
        h for h, n in token_prefixes("model/8", list(range(129)))
    }


def test_rust_observation_does_not_credit_descendants_of_evicted_nodes():
    rows = [(2, 1, "uid", list(range(64, 128)), True, 2)]
    assert rust_prefixes(cache(rows), {"uid": "model/7"}) == []


def test_rust_observation_hashes_across_unaligned_node_splits():
    rows = [
        (1, 0, "uid", list(range(37)), False, 1),
        (2, 1, "uid", list(range(37, 128)), True, 2),
    ]
    expected = token_prefixes("model/7", list(range(129)))[-1][0]
    assert rust_prefixes(cache(rows), {"uid": "model/7"}) == [expected]


def test_lora_admissions_exclude_base_probes_and_count_chunked_requests_once(
    monkeypatch,
):
    monkeypatch.setattr(observer, "counters", Counter())
    monkeypatch.setattr(observer, "seen_requests", set())
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True, is_decode=lambda: False),
        reqs=[
            SimpleNamespace(rid="probe", lora_id=None),
            SimpleNamespace(rid="first", lora_id="hot"),
            SimpleNamespace(rid="second", lora_id="cold"),
        ],
    )
    manager = SimpleNamespace(
        memory_pool=SimpleNamespace(uid_to_buffer_id={None: 0, "hot": 1})
    )
    observer.batch(batch, manager)
    manager.memory_pool.uid_to_buffer_id["cold"] = 2
    observer.batch(batch, manager)
    assert observer.counters == {
        "base_request_admissions": 1,
        "request_lora_hits": 1,
        "request_lora_misses": 1,
    }
