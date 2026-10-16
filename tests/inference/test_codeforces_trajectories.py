from types import SimpleNamespace
import math

from scripts.benchmarks.rl_rust import codeforces
from scripts.benchmarks.lora_routing.client import datum


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 99

    def apply_chat_template(self, messages, **kwargs):
        assert "tests_passed" not in messages[0]["content"]
        assert "passed 1 of 2 tests" in messages[0]["content"]
        return "feedback"

    def encode(self, text, **kwargs):
        return [12, 13]

    def decode(self, tokens, **kwargs):
        return "</think>print(1)"


def test_continuation_preserves_sampled_prefix_and_masks_feedback():
    prompt = codeforces.continuation(
        Tokenizer(), [1, 2], [3, 99], dict(tests_passed=1, tests_total=2)
    )
    assert prompt == [1, 2, 3, 99, 12, 13]
    row = datum(prompt, [20, 21], [-0.3, -0.4], 1.0, 4)
    assert row.model_input.to_ints() == [1, 2, 3, 99, 12, 13, 20]
    values = row.loss_fn_inputs["advantages"].data
    assert values == [0.0] * 5 + [0.25, 0.25]


def test_both_turns_use_same_sampler_and_real_judge_results(monkeypatch):
    calls = []
    judged = []

    class Sampler:
        def sample(self, prompt, count, params):
            calls.append((prompt.to_ints(), count))
            response = SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        tokens=[3, 99], logprobs=[-0.1, -0.2], stop_reason="stop"
                    )
                    for _ in range(count)
                ],
                prompt_cache_hit_tokens=0,
            )
            return SimpleNamespace(result=lambda **kw: response)

    async def judge(code, tests, app):
        judged.append(code)
        return dict(passed=False, tests_passed=1, tests_total=2, tests_run=2)

    monkeypatch.setattr(codeforces, "judge", judge)
    monkeypatch.setattr(codeforces.modal.App, "lookup", lambda *a, **kw: object())
    client = SimpleNamespace(
        cfg=dict(
            judge_app="test",
            samples_per_group=2,
            max_tokens=4,
            revision_max_tokens=2,
            context_tokens=20,
            temperature=1.0,
            top_p=1.0,
            seed=42,
        ),
        rows=[dict(tests=[{}, {}])],
        prompts=[[1, 2]],
        tokenizer=Tokenizer(),
        stops=[99],
        event=lambda *a, **kw: None,
    )
    group = codeforces.sample_group(client, 0, (Sampler(), 7), 0)
    assert group["policy_version"] == 7
    assert len(calls) == 3 and len(judged) == 4
    assert calls[0] == ([1, 2], 2)
    assert all(prompt[:4] == [1, 2, 3, 99] for prompt, n in calls[1:])
    assert all(
        len(s["turns"]) == 2 and math.isfinite(s["reward"]) for s in group["sequences"]
    )
