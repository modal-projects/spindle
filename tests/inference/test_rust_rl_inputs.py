import json
import pytest
from scripts.benchmarks.rl_rust.run import validate_inputs


def test_paired_manifest_check_waits_for_both_then_rejects_changed_inputs(tmp_path):
    manifest = dict(
        config_sha256="a",
        prompts_sha256="b",
        dataset_sha256="c",
        client_index=0,
        problem_offset=0,
        stop_token_ids=[99],
        packages={},
        source_sha256={},
    )
    policies = ["default", "load_balanced", "lora_radix"]
    (tmp_path / "schedule.json").write_text(json.dumps({"pairs": [policies]}))
    for policy in policies:
        path = tmp_path / policy / "measured/client-00/manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))
        validate_inputs(tmp_path)
    manifest["dataset_sha256"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Mismatched paired inputs"):
        validate_inputs(tmp_path)
