"""Token-preserving bash rollouts and exact sampler-distribution replay."""

import asyncio
import json
import re
import time
import uuid

from spindle.inference.sampling import sample_task
from spindle.providers.modal.fft_pool import proxy_auth_headers

from .config import (
    MAX_TOKENS_PER_TURN,
    MAX_TURNS,
    SAMPLING_TIMEOUT_SECONDS,
    TEMPERATURE,
    TOP_P,
    config,
)
from .environment import capture_patch, grade_patch, setup

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finish the coding task and submit the repository changes for grading.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
SYSTEM = (
    "Solve the user's coding task by inspecting and editing the repository with the bash tool. "
    "Run relevant tests. When finished, call the submit tool. "
    "Each tool invocation must use the advertised tool-call format."
)


def commands(text):
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(body)
            if call["name"] == "submit":
                calls.append({"command": "submit"})
                continue
            if call["name"] != "bash":
                continue
            arguments = call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(arguments)
        except (ValueError, KeyError, TypeError):
            # Qwen3.6 also supports the function/parameter XML representation.
            if "<function=submit>" in body:
                calls.append({"command": "submit"})
                continue
            command = re.search(
                r"<parameter=command>(.*?)</parameter>", body, re.DOTALL
            )
            if "<function=bash>" in body and command:
                calls.append({"command": command[1]})
    return [
        c for c in calls if isinstance(c, dict) and isinstance(c.get("command"), str)
    ]


def append_observation(row, tokens):
    row["tokens"].extend(tokens)
    row["active"].extend([0] * len(tokens))
    row["logq"].extend([0.0] * len(tokens))
    row["offsets"].extend([len(row["support"])] * len(tokens))


def append_generation(row, sequence):
    replay = sequence["replay"]
    start = len(row["support"])
    row["tokens"].extend(sequence["tokens"])
    row["active"].extend([1] * len(sequence["tokens"]))
    row["logq"].extend(replay["sampling_logprobs"])
    row["offsets"].extend(start + n for n in replay["sampling_mask_offsets"][1:])
    row["support"].extend(replay["sampling_mask_ids"])
    row["support_logq"].extend(replay["sampling_mask_logprobs"])


async def rollout(task, gateway, group, model_id, version, seed, tokenizer):
    started = time.time()
    session = str(uuid.uuid4())
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["problem_statement"]},
        ],
        tools=TOOLS,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    row = {
        "tokens": [],
        "active": [],
        "logq": [],
        "offsets": [0],
        "support": [],
        "support_logq": [],
        "versions": [],
        "instance_id": task["instance_id"],
        "group": group,
        "started_at": started,
        "requested_version": version,
        "format_error": False,
        "stop_reason": "max_turns",
    }
    append_observation(row, prompt)
    environment, baseline = await asyncio.to_thread(setup, task)
    consecutive_non_tool = 0
    try:
        for turn in range(MAX_TURNS):
            if len(row["tokens"]) >= config.max_context_length - 1:
                row["stop_reason"] = "context_limit"
                break
            stats = {}
            result = await sample_task(
                {
                    "request_id": f"{session}-{turn}",
                    "model_id": model_id,
                    "publish_version": version,
                    "latest": True,
                    "payload": {
                        "prompt": {
                            "chunks": [
                                {"type": "encoded_text", "tokens": row["tokens"]}
                            ]
                        },
                        "sampling_params": {
                            "temperature": TEMPERATURE,
                            "top_p": TOP_P,
                            "max_tokens": min(
                                MAX_TOKENS_PER_TURN,
                                config.max_context_length - len(row["tokens"]) - 1,
                            ),
                            "top_k": -1,
                            "seed": seed + turn,
                        },
                        "return_sampling_probabilities": True,
                    },
                },
                gateway,
                timeout=SAMPLING_TIMEOUT_SECONDS,
                retry_timeout=SAMPLING_TIMEOUT_SECONDS,
                headers=proxy_auth_headers(),
                session_id=session,
                context_length=config.max_context_length,
                stats=stats,
            )
            sequence = result["sequences"][0]
            append_generation(row, sequence)
            row["versions"].append(
                {
                    "start": stats["version_served_start"],
                    "end": stats["version_served_end"],
                    "at": time.time(),
                }
            )
            text = tokenizer.decode(sequence["tokens"], skip_special_tokens=False)
            calls = commands(text)
            print(
                json.dumps(
                    {
                        "event": "rollout_turn",
                        "instance_id": task["instance_id"],
                        "group": group,
                        "turn": turn + 1,
                        "generated_tokens": len(sequence["tokens"]),
                        "commands": [c["command"][:200] for c in calls],
                    }
                ),
                flush=True,
            )
            consecutive_non_tool = 0 if calls else consecutive_non_tool + 1
            if consecutive_non_tool >= 3:
                row["format_error"], row["stop_reason"] = True, "format_error"
                break
            observations = []
            submitted = False
            for call in calls:
                if call["command"].strip() == "submit":
                    submitted = True
                    break
                output = await asyncio.to_thread(
                    environment.execute,
                    call["command"],
                    cwd=task["cwd"],
                    timeout=min(300, max(1, int(call.get("timeout", 120)))),
                )
                observations.append(json.dumps(output, ensure_ascii=False))
            if submitted:
                row["stop_reason"] = "submit"
                break
            if not calls:
                observations = [
                    "Continue using the bash tool. Call the submit tool when finished."
                ]
            # Only tokenize newly observed text. Existing assistant token IDs,
            # including reasoning, remain exactly as sampled on every turn.
            observation = (
                "<|im_start|>user\n"
                + "\n".join(
                    f"<tool_response>\n{o}\n</tool_response>" for o in observations
                )
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
            # A generation stopped by length needs its assistant boundary closed.
            if sequence["stop_reason"] == "length":
                observation = "<|im_end|>\n" + observation
            added = tokenizer.encode(observation, add_special_tokens=False)
            remaining = config.max_context_length - len(row["tokens"]) - 1
            if len(added) > remaining:
                row["stop_reason"] = "context_limit"
                break
            append_observation(row, added)
        patch = await asyncio.to_thread(capture_patch, environment, task, baseline)
    finally:
        await asyncio.to_thread(environment.cleanup)
    reward, details = await asyncio.to_thread(grade_patch, task, patch)
    row.update(
        reward=reward,
        verifier=details,
        finished_at=time.time(),
        turns=turn + 1,
        generated_tokens=sum(row["active"]),
    )
    return row
