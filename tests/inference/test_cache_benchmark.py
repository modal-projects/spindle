import json
from pathlib import Path
import runpy
from types import SimpleNamespace


BENCHMARK = Path(__file__).resolve().parents[2] / "scripts/benchmarks/lora_routing"


def test_gpu_hit_counters_count_each_request_once_and_exclude_base(
    tmp_path, monkeypatch
):
    code = runpy.run_path(str(BENCHMARK / "cache_instrumentation.py"))["METRICS_MODULE"]
    code = code.replace(
        "/tmp/spindle-routing-metrics.json", str(tmp_path / "metrics.json")
    )
    metrics = {}
    exec(code, metrics)
    requests = [
        SimpleNamespace(rid="a", lora_id="hot"),
        SimpleNamespace(rid="b", lora_id="cold"),
        SimpleNamespace(rid="c", lora_id=None),
    ]
    metrics["requests"](requests, {"hot"})
    metrics["requests"](requests, {"hot", "cold"})
    counts = metrics["counts"]
    assert counts["request_lora_hits"] == counts["request_lora_misses"] == 1
    refs = {uid: SimpleNamespace(lora_name=uid + "/v1") for uid in ("hot", "cold")}
    metrics["loaded"]({None, "hot"}, {"hot", "cold"}, refs, 0.2)
    assert counts["gpu_loads"] == 1
    assert counts["gpu_evictions"] == 0  # The base-model slot is not a LoRA.
    metrics["loaded"]({"hot", "cold"}, {"hot"}, refs, 0.01)
    assert counts["gpu_evictions"] == 1
    state = json.loads((tmp_path / "metrics.json").read_text())
    assert state["resident_adapters"] == ["hot/v1"]


def test_trace_has_exact_lengths_repeated_prefixes_and_distinct_versions():
    make_trace = runpy.run_path(str(BENCHMARK / "cache_replay.py"))["make_trace"]
    clients = [{"model_id": "m", "versions": [4, 7], "prompts": [list(range(100))]}]
    kwargs = dict(
        streams=2, turns=3, prompt_tokens=64, shared_tokens=48, decode_tokens=16
    )
    left = make_trace(clients, **kwargs)
    assert left == make_trace(clients, **kwargs)
    assert len(left) == 4
    assert {s["version"] for s in left} == {4, 7}
    prefixes = set()
    for stream in left:
        assert all(len(p) == 64 for p in stream["prompts"])
        assert all(p[:48] == stream["prompts"][0][:48] for p in stream["prompts"])
        assert len({tuple(p[48:]) for p in stream["prompts"]}) == 3
        prefixes.add(tuple(stream["prompts"][0][:48]))
    assert len(prefixes) == 4


def test_concurrent_streams_share_only_their_conversation_prefix():
    make_trace = runpy.run_path(str(BENCHMARK / "cache_replay.py"))["make_trace"]
    clients = [{"model_id": "m", "versions": [4, 7], "prompts": [list(range(100))]}]
    trace = make_trace(
        clients, streams=4, conversations=2, turns=2, prompt_tokens=64, shared_tokens=48
    )
    groups = {}
    for stream in trace:
        key = (stream["version"], stream["conversation"])
        groups.setdefault(key, []).extend(stream["prompts"])
    assert len(groups) == 4
    for prompts in groups.values():
        assert len(prompts) == 4
        assert len({tuple(p[:48]) for p in prompts}) == 1
        assert len({tuple(p[48:]) for p in prompts}) == 4
