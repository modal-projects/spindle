# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///
"""Run the tinker-cookbook DPO recipe against Spindle and compare with Tinker.

DPO needs no Spindle-specific config: the recipe uses `forward_backward_custom`
(a client-side `forward` plus weighted `cross_entropy`) and computes reference
logprobs with `compute_logprobs` on a sampler published from the step-0 weights.
This runs the cookbook README's example command on the `qwen35-9b-lora-16k`
deployment and prints the last step next to the README's Tinker numbers.

    export TINKER_BASE_URL=https://your-modal-server-url
    export TINKER_API_KEY=...
    uv run scripts/e2e_dpo_qwen3_5_9b_lora.py --steps 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from tinker_cookbook.recipes.preference.dpo.train import CLIConfig, cli_main

# Step 49 of the cookbook DPO README (hhh, Qwen3.5-9B-Base, lr 1e-5, beta 0.1, batch 256).
TINKER_STEP_49 = {
    "dpo_loss": 0.690734,
    "accuracy": 0.515748,
    "margin": 0.005681,
    "chosen_reward": 0.008626,
    "rejected_reward": 0.002946,
    "time/step": 5.270600,
    "time/get_ref_logprobs": 2.125185,
}


def read_metrics(log_path: Path) -> list[dict]:
    for path in sorted(log_path.rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if any("dpo_loss" in row for row in rows):
            return [row for row in rows if "dpo_loss" in row]
    raise FileNotFoundError(f"no DPO metrics under {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--dataset", default="hhh")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-path", default="/tmp/spindle-dpo")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    cli_main(
        CLIConfig(
            model_name=args.base_model,
            dataset=args.dataset,
            renderer_name="role_colon",
            learning_rate=args.learning_rate,
            dpo_beta=args.dpo_beta,
            batch_size=args.batch_size,
            max_steps=args.steps,
            log_path=args.log_path,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            base_url=os.environ["TINKER_BASE_URL"],
            behavior_if_log_dir_exists="delete",
        )
    )

    rows = read_metrics(Path(args.log_path))
    first, last = rows[0], rows[-1]
    print(
        f"\n{'metric':<24}{'step 0':>12}{f'step {len(rows) - 1}':>12}{'Tinker 49':>12}"
    )
    for key, reference in TINKER_STEP_49.items():
        print(
            f"{key:<24}{first.get(key, math.nan):>12.6f}{last.get(key, math.nan):>12.6f}{reference:>12.6f}"
        )

    # At this learning rate DPO moves slowly; the README run ends just under ln 2.
    learned = last["dpo_loss"] < math.log(2) and last["margin"] > 0
    print(
        f"\nDPO {'learned' if learned else 'did NOT learn'}: final dpo_loss {last['dpo_loss']:.6f} vs ln2 {math.log(2):.6f}"
    )
    raise SystemExit(0 if learned else 1)


if __name__ == "__main__":
    main()
