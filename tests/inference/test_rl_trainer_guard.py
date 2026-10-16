"""A lost serving trainer requests scoped recovery, never experiment-wide teardown."""

import json
import pytest
from scripts.benchmarks.rl_four_arm.guard import inspect


@pytest.mark.parametrize("finished", [0, 1792118001.045022])
def test_guard_ignores_spare_and_recovers_serving_trainer(tmp_path, finished):
    directory = tmp_path / "default-no-split-k/measured/client-00"
    directory.mkdir(parents=True)
    (directory / "commit-001.json").write_text('{"update":1}')
    (tmp_path / "gpu-lifetimes.json").write_text(
        json.dumps(
            [
                dict(task_id="serving", gpu_count=8, finished_at=finished),
                dict(task_id="spare", gpu_count=8, finished_at=1792117966.817316),
            ]
        )
    )
    counts = inspect(tmp_path, {"default-no-split-k": "serving"})
    assert counts == {"default-no-split-k": 1}
    notice = directory.parent / "RECOVER_TRAINER.json"
    assert notice.exists() == bool(finished)
    assert not (tmp_path / "STOP").exists()
    if finished:
        original = notice.read_text()
        inspect(tmp_path, {"default-no-split-k": "serving"})
        assert notice.read_text() == original
