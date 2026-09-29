#!/usr/bin/env bash
set -e

# Run from the repository root so the Python config paths below work from any directory.
cd "$(dirname "$0")/.."

# Add a model by creating its Python config in src/spindle/configs/ and adding it here.
# Put the preferred recipe first when multiple configs share a model.
deployment_files=(
  src/spindle/configs/qwen35_9b_lora_16k.py
  src/spindle/configs/qwen35_9b_lora_64k.py
  src/spindle/configs/qwen35_4b_fft_64k.py
  src/spindle/configs/gpt_oss_20b_lora_64k.py
  src/spindle/configs/qwen36_35b_a3b_lora_32k.py
)

spindle deploy "${deployment_files[@]}" "$@"
