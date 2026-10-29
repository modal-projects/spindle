import asyncio
import math
from types import SimpleNamespace

from mixed_samplers import trajectory
from mixed_samplers.train import datum
from mixed_samplers.trajectory import append_generation, append_observation, commands


def test_multiturn_replay_aligns_targets_without_retokenizing_assistant():
    row = {
        "tokens": [],
        "active": [],
        "logq": [],
        "offsets": [0],
        "support": [],
        "support_logq": [],
    }
    append_observation(row, [10, 11])
    sequence = {
        "tokens": [20, 21],
        "replay": {
            "sampling_logprobs": [math.log(0.6), 0.0],
            "sampling_mask_offsets": [0, 2, 3],
            "sampling_mask_ids": [20, 22, 21],
            "sampling_mask_logprobs": [math.log(0.6), math.log(0.4), 0.0],
        },
    }
    append_generation(row, sequence)
    append_observation(row, [30, 31, 32])
    append_generation(row, sequence)
    result = datum(row, 2.0, 4)
    assert result.model_input.to_ints() == [10, 11, 20, 21, 30, 31, 32, 20]
    inputs = result.loss_fn_inputs
    assert inputs["target_tokens"].data == [11, 20, 21, 30, 31, 32, 20, 21]
    assert inputs["weights"].data == [0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25]
    assert inputs["sampling_mask_offsets"].data == [0, 0, 2, 3, 3, 3, 3, 5, 6]
    assert inputs["sampling_mask_ids"].data == [20, 22, 21, 20, 22, 21]


def test_qwen_xml_and_json_bash_calls():
    assert commands(
        "<tool_call>\n<function=bash>\n<parameter=command>pwd</parameter>\n</function>\n</tool_call>"
    ) == [{"command": "pwd"}]
    assert commands(
        '<tool_call>{"name":"bash","arguments":{"command":"submit"}}</tool_call>'
    ) == [{"command": "submit"}]
    assert commands("I am finished.") == []
    assert commands("<tool_call><function=submit>\n</function></tool_call>") == [
        {"command": "submit"}
    ]
    assert commands('<tool_call>{"name":"submit","arguments":{}}</tool_call>') == [
        {"command": "submit"}
    ]


def test_context_boundary_grades_existing_work_without_truncating_replay(monkeypatch):
    # SGLang accepts inputs strictly below context_len - 6.
    monkeypatch.setattr(trajectory, "config", SimpleNamespace(max_context_length=64))
    monkeypatch.setattr(trajectory, "proxy_auth_headers", dict)
    cleaned, graded, prompts = [], [], []
    environment = SimpleNamespace(cleanup=lambda: cleaned.append(True))
    monkeypatch.setattr(trajectory, "setup", lambda task: (environment, "base"))
    monkeypatch.setattr(trajectory, "execute_command", lambda *a, **k: {})
    monkeypatch.setattr(trajectory, "capture_patch", lambda *a: b"existing patch")

    def grade(task, patch):
        graded.append(patch)
        return 1.0, {"verifier_returncode": 0}

    monkeypatch.setattr(trajectory, "grade_patch", grade)

    async def sample(request, *a, stats, **k):
        tokens = request["payload"]["prompt"]["chunks"][0]["tokens"]
        prompts.append(list(tokens))
        assert len(tokens) < 58, "SGLang would reject this request"
        stats.update(version_served_start=1, version_served_end=1)
        return {
            "sequences": [
                {
                    "tokens": [42],
                    "stop_reason": "stop",
                    "replay": {
                        "sampling_logprobs": [0.0],
                        "sampling_mask_offsets": [0, 1],
                        "sampling_mask_ids": [42],
                        "sampling_mask_logprobs": [0.0],
                    },
                }
            ]
        }

    monkeypatch.setattr(trajectory, "sample_task", sample)
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **k: [10] * 57,
        decode=lambda *a, **k: '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
        encode=lambda *a, **k: [],
    )
    row = asyncio.run(
        trajectory.rollout(
            {"instance_id": "task", "problem_statement": "solve", "cwd": "/repo"},
            "gateway",
            "b200-nvfp4",
            "model",
            1,
            42,
            tokenizer,
        )
    )
    assert prompts == [[10] * 57]
    assert row["tokens"] == [10] * 57 + [42]
    assert row["active"] == [0] * 57 + [1]
    assert row["support"] == [42]
    assert row["stop_reason"] == "context_limit"
    assert row["reward"] == 1.0
    assert graded == [b"existing patch"]
    assert cleaned == [True]
