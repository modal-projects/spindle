"""Time CPU replay preparation; run unchanged against both revisions."""

import argparse
import cProfile
import gc
import json
import pstats
import time
from pathlib import Path
import numpy as np
from tinker import types
from spindle.replay import SamplingReplay
from spindle.backends.contract import ForwardBatch, ForwardItem
from spindle.backends.miles_runtime.data import prepare_batch
from spindle.backends.miles_runtime.replay_data import (
    replay_row,
    add_replay_to_train_data,
)
from spindle.backends.score_centering import validate_support

p = argparse.ArgumentParser()
p.add_argument("output")
p.add_argument("--profile", action="store_true")
args = p.parse_args()


def make(prompt, response, width):
    lengths = np.full(response, width, dtype=np.int64)
    # Ragged supports with normalized probabilities and the sampled target 0.
    lengths[::3] = np.maximum(1, lengths[::3] // 2)
    offsets = np.r_[0, np.cumsum(lengths)]
    rng = np.random.default_rng(0)
    ids = np.concatenate([rng.permutation(k) for k in lengths])
    logq = np.repeat(-np.log(lengths), lengths).astype(np.float32)
    old = (-np.log(lengths)).astype(np.float32)
    replay = SamplingReplay(
        prompt_tokens=prompt,
        temperature=1,
        sampling_mask_ids=ids.tolist(),
        sampling_mask_offsets=offsets.tolist(),
        sampling_mask_logprobs=logq.tolist(),
        sampling_logprobs=old.tolist(),
    )
    n = prompt + response - 1
    weights = np.r_[np.zeros(prompt - 1), np.ones(response)].astype(np.float32)
    return replay, n, weights


cases = [
    ("top_p_long_prompt", 49152, 256, 512),
    ("top_p_wide_support", 128, 8192, 512),
    ("singleton_support", 128, 8192, 1),
    ("small_support", 128, 8192, 4),
]
report = {}
for name, prompt, response, width in cases:
    replay, n, weights = make(prompt, response, width)
    runs = []
    last = None
    for _ in range(3):
        gc.collect()
        times = {}

        def timed(label, fn):
            t = time.perf_counter()
            out = fn()
            times[label] = time.perf_counter() - t
            return out

        inp = timed("training_inputs", lambda: replay.training_inputs(response))
        inp.update(
            target_tokens=types.TensorData(
                data=np.zeros(n, dtype=np.int64), dtype="int64", shape=[n]
            ),
            weights=types.TensorData(data=weights, dtype="float32", shape=[n]),
            advantages=types.TensorData(data=weights, dtype="float32", shape=[n]),
        )
        targets = np.zeros(n, dtype=np.int64)
        timed("replay_row", lambda: replay_row(inp, targets))
        timed(
            "validate_support",
            lambda: validate_support(inp, targets, weights, centered=True),
        )
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1] * n), loss_fn_inputs=inp
        )
        batch = ForwardBatch(
            items=(ForwardItem("m", (datum,)),), loss_fn="score_centered_mis"
        )
        prepared = timed(
            "prepare_batch",
            lambda: prepare_batch(batch, {"m": 0}, sequence_alignment=8),
        )
        data = {}
        timed(
            "handoff",
            lambda: add_replay_to_train_data(
                data, [row for _, row in prepared.slot_rows]
            ),
        )
        assert np.array_equal(
            data["rollout_sampling_mask_ids"][0],
            np.asarray(replay.sampling_mask_ids, dtype=np.int32),
        )
        assert np.array_equal(
            data["rollout_sampling_mask_logprobs"][0],
            np.asarray(replay.sampling_mask_logprobs, dtype=np.float32),
        )
        assert data["rollout_sampling_mask_offsets"][0][-1] == len(
            replay.sampling_mask_ids
        )
        runs.append(times)
        last = batch
    report[name] = {
        "input_positions": n,
        "support_entries": len(replay.sampling_mask_ids),
        "runs": runs,
        "median_s": {k: float(np.median([r[k] for r in runs])) for k in runs[0]},
    }
    if args.profile and name == "top_p_wide_support":
        prof = cProfile.Profile()
        prof.enable()
        prepare_batch(last, {"m": 0})
        prof.disable()
        pstats.Stats(prof).strip_dirs().sort_stats("cumulative").print_stats(18)
    replay = inp = last = data = prepared = batch = None
    gc.collect()
plain = types.Datum(
    model_input=types.ModelInput.from_ints([1] * 4096),
    loss_fn_inputs={
        "target_tokens": types.TensorData(
            data=np.ones(4096, dtype=np.int64), dtype="int64", shape=[4096]
        ),
        "weights": types.TensorData(
            data=np.ones(4096, dtype=np.float32), dtype="float32", shape=[4096]
        ),
    },
)
batch = ForwardBatch(items=(ForwardItem("m", (plain,) * 32),), loss_fn="cross_entropy")
t = []
for _ in range(3):
    start = time.perf_counter()
    prepare_batch(batch, {"m": 0})
    t.append(time.perf_counter() - start)
report["plain_ce_32x4k"] = {"runs": t, "median_s": float(np.median(t))}
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
