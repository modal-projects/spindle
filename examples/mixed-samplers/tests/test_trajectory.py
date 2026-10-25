import math

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
