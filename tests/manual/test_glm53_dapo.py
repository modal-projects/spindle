"""CPU checks; run with scripts and the pinned Miles checkout on PYTHONPATH."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("miles.rollout.rm_hub.math_dapo_utils")

from glm53_dapo_data import (
    grade_sample,
    group_advantages,
    sample_problems,
    training_data,
)


def response(answer="34", *, truncated=False):
    return {
        "text": f"Calculation.\nAnswer: \\boxed{{{answer}}}",
        "output_ids": [7, 8],
        "meta_info": {
            "finish_reason": {"type": "length" if truncated else "stop"},
            "output_token_logprobs": [[-0.2, 7, None], [-0.3, 8, None]],
            "weight_version_start": 3,
        },
    }


def test_dataset_reward_and_token_alignment():
    assert grade_sample(response(), {"answer": "34"})["reward"] == 1
    assert grade_sample(response("35"), {"answer": "34"})["reward"] == -1
    bad = response()
    bad["meta_info"]["output_token_logprobs"][0][1] = 99
    with pytest.raises(AssertionError, match="do not match"):
        grade_sample(bad, {"answer": "34"})


def test_incomplete_responses_do_not_create_a_training_signal():
    samples = [
        {"reward": 1.0, "truncated": False},
        {"reward": 1.0, "truncated": False},
        {"reward": -1.0, "truncated": True},
    ]
    np.testing.assert_array_equal(group_advantages(samples), [0, 0, 0])
    samples[1]["reward"] = -1.0
    advantages = group_advantages(samples)
    assert advantages[2] == 0
    assert advantages[0] == pytest.approx(-advantages[1])
    assert np.std(advantages[:2], ddof=1) == pytest.approx(1.0, abs=1e-5)


def test_prompt_mask_and_global_token_normalization():
    good = grade_sample(response(), {"answer": "34"})
    bad = grade_sample(response("35"), {"answer": "34"})
    bad.update(tokens=[7, 8, 9, 10], logprobs=[-0.2, -0.3, -0.4, -0.5])
    truncated = grade_sample(response(truncated=True), {"answer": "34"})
    groups = [{"problem": {"tokens": [1, 2, 3]}, "samples": [good, bad, truncated]}]
    data, replay, total = training_data(groups)
    assert total == 6  # Excludes the two incomplete-response tokens.
    assert data[0].model_input.to_ints() == [1, 2, 3, 7]
    assert data[0].loss_fn_inputs["target_tokens"].data == [2, 3, 7, 8]
    advantages = group_advantages(groups[0]["samples"])
    for datum, sample, advantage in zip(
        data, groups[0]["samples"], advantages, strict=True
    ):
        weights = datum.loss_fn_inputs["advantages"].data
        assert weights[:2] == [0, 0]
        np.testing.assert_allclose(weights[2:], advantage / total)
        np.testing.assert_allclose(
            datum.loss_fn_inputs["logprobs"].data[2:], sample["logprobs"], rtol=1e-7
        )
    assert replay[1] == (2, bad["logprobs"])


def test_mixed_rollouts_pin_each_clients_adapter():
    received = []

    def generate(operation, payload):
        assert operation == "generate"
        received.append(payload)
        return response()

    config = {
        "run_id": "experiment",
        "group_size": 2,
        "seed": 42,
        "max_tokens": 8192,
        "concurrency": 4,
    }
    sampler = SimpleNamespace(request=SimpleNamespace(remote=generate))
    problem = {"id": 5, "tokens": [1, 2], "answer": "34"}
    groups = sample_problems(sampler, [(0, problem), (1, problem)], config, 3)
    assert [g["client"] for g in groups] == [0, 1]
    assert {p["weight_run_id"] for p in received} == {
        "experiment-client0",
        "experiment-client1",
    }
    assert all(p["weight_version"] == {"exact_version": 3} for p in received)
    assert len({p["sampling_params"]["sampling_seed"] for p in received}) == 4
    with pytest.raises(AssertionError, match="wrong policy"):
        sample_problems(sampler, [(0, problem)], config, 4)
