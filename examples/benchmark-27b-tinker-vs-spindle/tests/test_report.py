import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_report", ROOT / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_measured_tinker_baseline():
    result = report.summarize_tinker(ROOT / "reference/tinker-events.jsonl", 4, 9)
    assert result["output_tokens"] == 4_628_189
    assert result["seconds"] == pytest.approx(1272.714908838272)
    assert result["aggregate_output_tps"] == pytest.approx(3636.469540711665)
    assert result["cost_per_million_output_tokens"] == pytest.approx(
        [9.768049958633927, 9.791455744352705]
    )


def test_tinker_rejects_missing_update_instead_of_underpricing(tmp_path):
    events = report.load_events(ROOT / "reference/tinker-events.jsonl")
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events if e["update"] != 5))
    with pytest.raises(ValueError, match="consecutive completed updates"):
        report.summarize_tinker(path, 4, 9)


def test_spindle_clips_tokens_and_gpu_cost_to_common_window(tmp_path):
    for index, start in enumerate([5, 10]):
        client = tmp_path / f"client-{index:02d}"
        client.mkdir()
        (client / "events.jsonl").write_text(
            json.dumps(dict(event="rollout_start", time=start)) + "\n"
        )
        # Include a group started before the window but finishing within it,
        # as the historical completed-group measurement does.
        for i, end in enumerate([9, 15, 21]):
            with gzip.open(client / f"group-{i:06d}.json.gz", "wt") as stream:
                json.dump(
                    dict(start=4, end=end, sequences=[dict(tokens=[1, 2, 3])]), stream
                )
    (tmp_path / "gpu-lifetimes.json").write_text(
        json.dumps(
            [
                dict(gpu_count=8, gpu_type="H200", started_at=1, finished_at=0),
                dict(gpu_count=8, gpu_type="H200", started_at=12, finished_at=18),
                dict(gpu_count=8, gpu_type="H200", started_at=1, finished_at=9),
            ]
        )
    )
    result = report.summarize_spindle(tmp_path, 20, expected_clients=2)
    assert result["seconds"] == 10
    assert result["output_tokens"] == 6
    assert result["aggregate_output_tps"] == 0.6
    assert result["mean_output_tps_per_client"] == 0.3
    assert result["gpu_seconds"] == 128
    assert result["gpu_cost_usd"] == pytest.approx(128 * 0.001261)
    with pytest.raises(ValueError, match="Incomplete run"):
        report.summarize_spindle(tmp_path, expected_clients=2)
    with pytest.raises(ValueError, match="Expected 8 Spindle clients"):
        report.summarize_spindle(tmp_path, 20)
