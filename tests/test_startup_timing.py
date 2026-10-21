import json

from spindle import startup_timing
from spindle.startup_timing import EVENT_NAME, StartupTimeline, timeline


def test_marks_measure_phases_from_origin(capsys) -> None:
    clock = StartupTimeline("trainer", origin=100.0, labels={"instance_id": "e1"})
    clock.mark("a")
    clock.mark("b", extra=1)

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["event"] for line in lines] == [EVENT_NAME, EVENT_NAME]
    assert [line["mark"] for line in lines] == ["a", "b"]
    assert lines[0]["component"] == "trainer"
    assert lines[0]["instance_id"] == "e1"
    assert lines[1]["extra"] == 1
    assert lines[0]["since_start_s"] == lines[0]["phase_s"]
    assert lines[1]["since_start_s"] >= lines[1]["phase_s"]

    phases = clock.phases()
    assert set(phases) == {"a", "b"}
    assert phases["a"] >= 0 and phases["b"] >= 0
    summary = clock.as_dict()
    assert summary["origin"] == 100.0
    assert set(summary["marks"]) == {"a", "b"}


def test_process_start_time_is_before_now() -> None:
    import time

    assert startup_timing.process_start_time() <= time.time()


def test_timeline_registry_is_per_component(monkeypatch) -> None:
    monkeypatch.setattr(startup_timing, "_TIMELINES", {})
    assert timeline("sampler") is timeline("sampler")
    assert timeline("sampler") is not timeline("trainer")
