"""Native Miles GPU scoring regression; launched by check_multi_targets_gpu.py."""

import argparse
import json
import os
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core.fusions import fused_cross_entropy
from miles.backends.training_utils.loss_hub import tinker_losses
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.tinker.runtime import _build_train_data
from miles.utils.ft_utils.process_group_utils import GroupInfo
from tinker import Datum, ModelInput, TensorData

# Import the production actor to install the exact CP/replay/multi-target hooks.
from spindle.backends.miles_runtime import actor  # noqa: F401
from spindle.backends.contract import ForwardBatch, ForwardItem
from spindle.backends.miles_runtime.data import build_outputs, prepare_batch


def initialize(tp, cp):
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    groups = {}
    for name, partitions in (
        ("tp", [list(range(c * tp, (c + 1) * tp)) for c in range(cp)]),
        ("cp", [[c * tp + t for c in range(cp)] for t in range(tp)]),
        ("single", [[r] for r in range(tp * cp)]),
    ):
        for ranks in partitions:
            group = dist.new_group(ranks)
            if rank in ranks:
                groups[name] = GroupInfo(
                    rank=ranks.index(rank), size=len(ranks), group=group
                )
    single = groups["single"]
    set_parallel_state(
        ParallelState(
            intra_dp=single,
            intra_dp_cp=groups["cp"],
            cp=groups["cp"],
            tp=groups["tp"],
            pp=single,
            ep=single,
            etp=single,
            indep_dp=single,
        )
    )
    return rank % tp, rank // tp


def make_datum(length, choices, final_only):
    candidates = torch.tensor([1, 257, 513, 769, 1023])[: choices or 1]
    targets = candidates.repeat(length, 1)
    weights = torch.linspace(-0.5, 1.0, targets.numel()).reshape(targets.shape)
    if final_only:
        weights[:-1] = 0
    if choices is None:
        targets, weights = targets[:, 0], weights[:, 0]
    return Datum(
        model_input=ModelInput.from_ints([3] * length),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=targets.flatten().tolist(),
                dtype="int64",
                shape=list(targets.shape),
            ),
            "weights": TensorData(
                data=weights.flatten().tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
        },
    )


def run_case(tp, cp, tp_rank, cp_rank, dtype, final_only, widths):
    request = ForwardBatch(
        items=(
            ForwardItem(
                "test",
                tuple(
                    make_datum(length, width, final_only)
                    for length, width in zip((1631, 7, 3), widths, strict=True)
                ),
            ),
        ),
        loss_fn="cross_entropy",
        loss_fn_config={},
    )
    prepared = prepare_batch(request, {"test": 0}, sequence_alignment=2 * tp * cp)
    assert len(prepared.slot_rows) == 3
    batch = _build_train_data(list(prepared.slot_rows))
    batch["unconcat_tokens"] = [
        torch.tensor(row["tokens"], device="cuda") for _, row in prepared.slot_rows
    ]
    batch["loss_fn_config"] = {}
    lengths = batch["total_lengths"]
    torch.manual_seed(109)
    full = torch.randn(sum(lengths), 1024, device="cuda", dtype=dtype)
    positions = []
    start = 0
    for length in lengths:
        if cp == 1:
            positions.extend(range(start, start + length))
        else:
            chunk = length // (2 * cp)
            for index in (cp_rank, 2 * cp - cp_rank - 1):
                positions.extend(
                    range(start + index * chunk, start + (index + 1) * chunk)
                )
        start += length
    positions = torch.tensor(positions, device="cuda")
    vocab = slice(tp_rank * (1024 // tp), (tp_rank + 1) * (1024 // tp))
    local = full[positions, vocab].unsqueeze(0).clone().requires_grad_()
    reference = full.detach().float().requires_grad_()
    args = argparse.Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=256,
        vocab_size=1024,
        debug_unified_grad_fused_logprob=False,
    )
    loss, output = tinker_losses.cross_entropy_loss_function(args, batch, local, None)
    expected_loss = 0
    expected_scores = []
    start = 0
    for (_, row), length in zip(prepared.slot_rows, lengths, strict=True):
        targets = torch.tensor(row["target_tokens"], device="cuda")
        weights = torch.tensor(row["weights"], device="cuda")
        scores = reference[start : start + length - 1].log_softmax(-1)
        selected = scores.gather(-1, targets if targets.ndim == 2 else targets[:, None])
        if targets.ndim == 1:
            selected = selected.squeeze(-1)
        expected_scores.append(selected)
        expected_loss -= (selected * weights).sum()
        start += length
    score_error = 0.0
    for record, expected in zip(output["per_datum"], expected_scores, strict=True):
        actual = record["logprobs"].cuda()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        score_error = max(score_error, (actual - expected).abs().max().item())
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-4)
    loss.backward()
    expected_loss.backward()
    # The pinned Megatron fused CE casts each column's gradient to BF16,
    # including when the logits are FP32. Compare against independent native
    # single-column calls as well as the unrounded dense reference.
    native_logits = local.detach().clone().requires_grad_()
    native_loss = 0
    for column in range(5):
        single = dict(batch)
        single["target_tokens"], single["loss_weights"] = [], []
        for targets, weights in zip(
            batch["target_tokens"], batch["loss_weights"], strict=True
        ):
            targets = torch.as_tensor(targets)
            weights = torch.as_tensor(weights)
            width = targets.shape[1] if targets.ndim == 2 else 1
            selected = min(column, width - 1)
            single["target_tokens"].append(
                targets[:, selected] if targets.ndim == 2 else targets
            )
            weight = weights[:, selected] if weights.ndim == 2 else weights
            single["loss_weights"].append(
                weight if column < width else torch.zeros_like(weight)
            )
        column_loss, _ = tinker_losses.cross_entropy_loss_function(
            args, single, native_logits, None
        )
        native_loss = native_loss + column_loss
    native_loss.backward()
    torch.testing.assert_close(loss, native_loss, rtol=2e-5, atol=2e-4)
    # Uniform label widths share one FP32 softmax, so only the native path rounds to BF16.
    rounded = dtype == torch.bfloat16 or len(set(widths)) == 1
    torch.testing.assert_close(
        local.grad,
        native_logits.grad,
        rtol=0.02 if rounded else 2e-5,
        atol=0.008 if rounded else 2e-6,
    )
    expected_grad = reference.grad[positions, vocab].unsqueeze(0).to(dtype)
    torch.testing.assert_close(
        local.grad,
        expected_grad,
        rtol=0.02,
        atol=0.008,
    )
    raw = [
        {"loss": record["loss"].item(), "logprobs": record["logprobs"].tolist()}
        for record in output["per_datum"]
    ]
    (result,) = build_outputs(request, prepared, raw)
    assert [record["logprobs"].shape for record in result.loss_fn_outputs] == [
        [length] if width is None else [length, width]
        for length, width in zip((1631, 7, 3), widths, strict=True)
    ]
    return {
        "tp": tp,
        "cp": cp,
        "widths": widths,
        "dtype": str(dtype),
        "final_only": final_only,
        "max_logprob_error": score_error,
        "max_native_gradient_error": (local.grad - native_logits.grad)
        .abs()
        .max()
        .item(),
        "max_gradient_error": (local.grad - expected_grad).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    config = parser.parse_args()
    tp_rank, cp_rank = initialize(config.tp, config.cp)
    if dist.get_rank() == 0:
        source = Path(fused_cross_entropy.__file__).read_text()
        print(
            "NATIVE_CE_BF16_BACKWARD",
            "grad_input.to(torch.bfloat16)" in source,
            flush=True,
        )
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for final_only, widths in (
                (True, (5, None, 1)),
                (False, (5, None, 1)),
                (False, (5, 5, 5)),
            ):
                result = run_case(
                    config.tp, config.cp, tp_rank, cp_rank, dtype, final_only, widths
                )
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, result)
                if dist.get_rank() == 0:
                    print("GPU_RESULT " + json.dumps(gathered), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
