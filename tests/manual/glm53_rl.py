"""Small real-rollout RL loop, invoked by validate_glm53_rl.py."""

import json
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np
import torch
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData

from spindle.backends.contract import ForwardBatch, ForwardItem, ModelSpec
from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_lora import MilesCommandBackend

PROBLEMS = (
    (
        "A shop buys 48 notebooks for $3 each. It sells 35 for $5 each and returns the rest to the supplier for $2 each. What is its total profit in dollars?",
        57,
    ),
    (
        "A tank is three-fifths full. Adding 30 liters makes it four-fifths full. What is the tank's capacity in liters?",
        150,
    ),
    (
        "A train travels 135 km in 90 minutes. It returns along the same route at 60 km per hour. How many minutes does the entire round trip take?",
        225,
    ),
    (
        "A club has 120 members. Three-tenths are adults and the rest are children. How many children are in the club?",
        84,
    ),
)


def reward(text, expected, length):
    answers = re.findall(r"\\boxed\{\s*(-?[\d,]+)\s*\}", text)
    correct = bool(answers) and int(answers[-1].replace(",", "")) == expected
    # Correctness dominates; a small length penalty gives concise solutions a
    # preference even when every member of an easy group answers correctly.
    return float(correct) - 0.01 * length / 1024, correct


def main(settings, sampler):
    torch.manual_seed(42)
    settings = dict(settings)
    settings["cli_options"] = {
        **settings["cli_options"],
        "global_batch_size": 8,
        "micro_batch_size": 1,
    }
    settings["hf_checkpoint"] = "/validation/model"
    run_id = "glm53-rl-" + uuid.uuid4().hex[:12]
    output = Path("/checkpoints") / run_id
    output.mkdir()
    started = time.monotonic()
    print("RL STARTUP", run_id, flush=True)
    backend = MilesCommandBackend(
        MilesBackendConfig(**settings),
        checkpoint_dir=Path("/checkpoints"),
        capture_dir=Path("/tmp/captures"),
        base_model="zai-org/GLM-5.3-Flash",
    )
    report = {"run_id": run_id, "startup_s": time.monotonic() - started, "steps": []}
    print("RL BACKEND READY", report["startup_s"], flush=True)
    try:
        backend.accept_model(
            run_id,
            ModelSpec(
                base_model="zai-org/GLM-5.3-Flash",
                parameterization="lora",
                lora_config=LoraConfig(rank=32, train_unembed=False),
            ),
        )
        prompts = [
            sampler.request.remote(
                "encode",
                question
                + " Show a short calculation, then give the final integer answer in \\boxed{}.",
            )
            for question, _ in PROBLEMS
        ]
        fixed_probe = prompts[0]
        probe_before = sampler.request.remote(
            "generate",
            {
                "input_ids": fixed_probe,
                "return_logprob": True,
                "logprob_start_len": 0,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
        )
        for step in range(2):
            step_started = time.monotonic()
            requests = []
            for group in range(2):
                problem = 2 * step + group
                for member in range(4):
                    payload = {
                        "input_ids": prompts[problem],
                        "return_logprob": True,
                        "sampling_params": {
                            "max_new_tokens": 1024,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "sampling_seed": 42 + 8 * step + 4 * group + member,
                        },
                    }
                    if step:
                        payload.update(
                            weight_run_id=run_id, weight_version={"exact_version": step}
                        )
                    requests.append(payload)
            print("RL SAMPLING", step + 1, flush=True)
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=8) as pool:
                samples = list(
                    pool.map(
                        lambda payload: sampler.request.remote("generate", payload),
                        requests,
                    )
                )
            sampling_s = time.monotonic() - start
            (output / f"samples-{step + 1}.json").write_text(json.dumps(samples))
            modal.Volume.from_name("spindle-glm53-pr26-rl-checkpoints").commit()
            print("RL SAMPLED", step + 1, sampling_s, flush=True)
            data, rewards, correctness, lengths, replay_logprobs = [], [], [], [], []
            for group in range(2):
                group_samples = samples[4 * group : 4 * group + 4]
                expected = PROBLEMS[2 * step + group][1]
                scores = [
                    reward(s["text"], expected, len(s["output_ids"]))
                    for s in group_samples
                ]
                group_rewards = np.array([value for value, _ in scores])
                advantages = (group_rewards - group_rewards.mean()) / (
                    group_rewards.std() + 1e-8
                )
                for member, (sample, advantage) in enumerate(
                    zip(group_samples, advantages, strict=True)
                ):
                    if step:
                        assert sample["meta_info"]["weight_version_start"] == step
                    generated = sample["output_ids"]
                    logprobs = [
                        row[0] for row in sample["meta_info"]["output_token_logprobs"]
                    ]
                    assert generated and len(generated) == len(logprobs)
                    assert all(math.isfinite(value) for value in logprobs)
                    prompt = prompts[2 * step + group]
                    tokens = prompt + generated
                    masked = len(prompt) - 1
                    sampling_lp = [0.0] * masked + logprobs
                    inputs = {
                        "target_tokens": TensorData(
                            data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]
                        ),
                        "logprobs": TensorData(
                            data=sampling_lp, dtype="float32", shape=[len(tokens) - 1]
                        ),
                        "advantages": TensorData(
                            data=[0.0] * masked + [float(advantage)] * len(generated),
                            dtype="float32",
                            shape=[len(tokens) - 1],
                        ),
                    }
                    data.append(Datum(ModelInput.from_ints(tokens[:-1]), inputs))
                    replay_logprobs.append((masked, logprobs))
                    rewards.append(scores[member][0])
                    correctness.append(scores[member][1])
                    lengths.append(len(generated))
            assert any(
                any(v != 0 for v in d.loss_fn_inputs["advantages"].data) for d in data
            ), "All sampled rewards were identical; no policy gradient"
            print("RL FORWARD BACKWARD", step + 1, lengths, flush=True)
            start = time.monotonic()
            (forward,) = backend.forward_backward(
                ForwardBatch(
                    items=(ForwardItem(run_id, tuple(data)),),
                    loss_fn="ppo",
                    loss_fn_config={
                        "clip_low_threshold": 0.8,
                        "clip_high_threshold": 1.2,
                    },
                )
            )
            forward_s = time.monotonic() - start
            differences = []
            for result, (masked, old) in zip(
                forward.loss_fn_outputs, replay_logprobs, strict=True
            ):
                current = np.array(result["logprobs"].data[masked:])
                assert (
                    current.shape == np.array(old).shape and np.isfinite(current).all()
                )
                differences.extend(np.abs(current - old).tolist())
            print(
                "RL LOGPROBS",
                step + 1,
                {"mean_abs": float(np.mean(differences)), "max_abs": max(differences)},
                flush=True,
            )
            start = time.monotonic()
            (optimizer,) = backend.optim_step((run_id,), AdamParams(learning_rate=1e-5))
            optimizer_s = time.monotonic() - start
            assert optimizer.metrics["update_successful:mean"] == 1
            assert (
                math.isfinite(optimizer.metrics["grad_norm:mean"])
                and optimizer.metrics["grad_norm:mean"] > 0
            )
            print("RL PUBLISHING", step + 1, optimizer.metrics, flush=True)
            start = time.monotonic()
            capture = f"{run_id}-{step + 1}"
            backend.capture_sampler_snapshot(run_id, capture, step + 1)
            capture_s = time.monotonic() - start
            start = time.monotonic()
            backend.publish_sampler_snapshot(capture)
            publication_s = time.monotonic() - start
            start = time.monotonic()
            probe_after = sampler.request.remote(
                "generate",
                {
                    "input_ids": fixed_probe,
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "weight_run_id": run_id,
                    "weight_version": {"exact_version": step + 1},
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                },
            )
            load_probe_s = time.monotonic() - start
            assert probe_after["meta_info"]["weight_version_start"] == step + 1
            before_lp = np.array(
                [x[0] for x in probe_before["meta_info"]["input_token_logprobs"][1:]]
            )
            after_lp = np.array(
                [x[0] for x in probe_after["meta_info"]["input_token_logprobs"][1:]]
            )
            assert before_lp.shape == after_lp.shape and np.isfinite(after_lp).all()
            delta = float(np.max(np.abs(after_lp - before_lp)))
            assert delta > 0, "Updated adapter did not change probe logprobs"
            row = {
                "step": step + 1,
                "reward_mean": float(np.mean(rewards)),
                "correct_fraction": float(np.mean(correctness)),
                "generation_length_mean": float(np.mean(lengths)),
                "generation_lengths": lengths,
                "sampling_s": sampling_s,
                "forward_backward_s": forward_s,
                "optimizer_s": optimizer_s,
                "capture_s": capture_s,
                "publication_s": publication_s,
                "load_and_probe_s": load_probe_s,
                "step_s": time.monotonic() - step_started,
                "train_sample_logprob_diff_mean": float(np.mean(differences)),
                "train_sample_logprob_diff_max": max(differences),
                "probe_logprob_change_max": delta,
                "optimizer": optimizer.metrics,
            }
            report["steps"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2))
            modal.Volume.from_name("spindle-glm53-pr26-rl-checkpoints").commit()
            print("RL STEP", json.dumps(row), flush=True)
            probe_before = probe_after
        return report
    finally:
        backend.close()
