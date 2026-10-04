# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "tinker>=0.24,<0.25",
#   "wandb",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///
"""Run the tinker-cookbook DPO recipe against Spindle.

DPO needs no Spindle-specific config: the recipe uses `forward_backward_custom`
(a client-side `forward` plus weighted `cross_entropy`) and computes reference
logprobs with `compute_logprobs` on a sampler published from the step-0 weights.
This runs the recipe on the `qwen35-9b-lora-16k` deployment. For the recipe and
Tinker's reference metrics, see
https://github.com/thinking-machines-lab/tinker-cookbook/blob/c8ed9c764b59161391156f980102d82f05014765/tinker_cookbook/recipes/preference/dpo/README.md

    export TINKER_BASE_URL=https://your-modal-server-url
    export TINKER_API_KEY=...
    uv run scripts/e2e_dpo_qwen3_5_9b_lora.py
"""

from __future__ import annotations

import argparse
import os

from tinker_cookbook.recipes.preference.dpo.train import CLIConfig, cli_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--dataset", default="hhh")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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


if __name__ == "__main__":
    main()
