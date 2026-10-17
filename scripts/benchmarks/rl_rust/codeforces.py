"""Two-turn Codeforces trajectories using the example's isolated judge and reward."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time

import modal
from tinker import types
from codegolf.judge import judge
from codegolf.prompts import THINKING_CODEGOLF_PROMPT
from codegolf.reward import extract_code, score

judge_slots = threading.Semaphore(8)  # Per training client, independent of generation.


def initial_messages(problem):
    return [
        dict(role="system", content=THINKING_CODEGOLF_PROMPT),
        dict(role="user", content=problem["statement"]),
    ]


def continuation(tokenizer, prompt, tokens, verdict):
    feedback = (
        f"Your program passed {verdict['tests_passed']} of {verdict['tests_total']} tests. "
        "Revise it to pass all tests and shorten the correct program. "
        "Return only the complete executable Python program."
    )
    # Preserve the exact previously sampled tokens, including reasoning. Re-encoding
    # assistant text can change token boundaries and destroy valid prefix reuse.
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    boundary = "\n" if tokens[-1] == end else "<|im_end|>\n"
    suffix = boundary + tokenizer.apply_chat_template(
        [dict(role="user", content=feedback)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt + tokens + tokenizer.encode(suffix, add_special_tokens=False)


def sequence_row(sequence, prompt, tokenizer):
    tokens, logprobs = list(sequence.tokens), list(sequence.logprobs or [])
    if (
        not tokens
        or len(tokens) != len(logprobs)
        or not all(map(math.isfinite, logprobs))
    ):
        raise ValueError("Invalid code trajectory tokens/logprobs")
    return dict(
        prompt=prompt,
        tokens=tokens,
        logprobs=logprobs,
        text=tokenizer.decode(tokens, skip_special_tokens=True),
        stop_reason=str(sequence.stop_reason),
    )


def sample_group(client, index, snapshot, row_index):
    sampler, version = snapshot
    cfg = client.cfg
    problem, prompt = client.rows[row_index], client.prompts[row_index]
    app = modal.App.lookup(cfg["judge_app"], create_if_missing=True)
    started = time.time()
    first = sampler.sample(
        types.ModelInput.from_ints(prompt),
        cfg["samples_per_group"],
        types.SamplingParams(
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            stop=client.stops,
            seed=cfg["seed"] + index,
        ),
    )
    client.event("group_submitted", group=index, policy_version=version)
    first = first.result(timeout=client.cfg.get("sampling_timeout_seconds", 3600))
    if len(first.sequences) != cfg["samples_per_group"]:
        raise ValueError("Incomplete initial Codeforces group")

    def verify(code):
        began = time.time()
        with judge_slots:
            result = asyncio.run(judge(code, problem["tests"], app))
        return result, time.time() - began

    def branch(pair):
        sibling, initial = pair
        turn1 = sequence_row(initial, prompt, client.tokenizer)
        verdict, judge_seconds = verify(
            extract_code(turn1["text"], require_thinking_end=True)
        )
        next_prompt = continuation(client.tokenizer, prompt, turn1["tokens"], verdict)
        budget = min(
            cfg["revision_max_tokens"], cfg["context_tokens"] - len(next_prompt) - 1
        )
        if budget <= 0:
            raise ValueError(
                "Codeforces trajectory exceeds context; never silently truncate history"
            )
        began = time.time()
        response = sampler.sample(
            types.ModelInput.from_ints(next_prompt),
            1,
            types.SamplingParams(
                max_tokens=budget,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                stop=client.stops,
                seed=cfg["seed"] + index * cfg["samples_per_group"] + sibling + 100000,
            ),
        ).result(timeout=client.cfg.get("sampling_timeout_seconds", 3600))
        if len(response.sequences) != 1:
            raise ValueError("Incomplete revision")
        turn2 = sequence_row(response.sequences[0], next_prompt, client.tokenizer)
        turn2.update(
            seconds=time.time() - began, cached_tokens=response.prompt_cache_hit_tokens
        )
        code = extract_code(turn2["text"])
        final, elapsed = verify(code)
        tokens = turn1["tokens"] + turn2["tokens"]
        return dict(
            tokens=tokens,
            logprobs=turn1["logprobs"] + turn2["logprobs"],
            turns=[turn1, turn2],
            first_verdict=verdict,
            final_verdict=final,
            judge_seconds=judge_seconds + elapsed,
            reward=score(
                final["passed"],
                code,
                scale=2048,
                bonus=0.30,
                output_tokens=len(tokens),
                token_penalty=0.20,
                token_scale=16384,
            ),
        )

    # Every sibling progresses independently after its own test feedback.
    with ThreadPoolExecutor(cfg["samples_per_group"]) as pool:
        sequences = list(pool.map(branch, enumerate(first.sequences)))
    return dict(
        group=index,
        row=row_index,
        policy_version=version,
        prompt=prompt,
        start=started,
        end=time.time(),
        sequences=sequences,
        first_sequence_cached_tokens=first.prompt_cache_hit_tokens,
    )
