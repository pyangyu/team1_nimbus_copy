#!/usr/bin/env bash
# Shared settings, sourced by the other scripts. Set any of these in your shell first to change them.
#   WORK      where traces, SFT data, adapters and eval results go   (default ~/brownout_train)
#   VENV      python venv with vllm, peft, datasets, math-verify     (default ~/venv)
#   MODEL     base model                                              (default Qwen/Qwen3-4B)
#   GPU_UTIL  share of the GPU vLLM may take; use 0.85 on a desktop card that also drives a screen
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/brownout_train}"
VENV="${VENV:-$HOME/venv}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
GPU_UTIL="${GPU_UTIL:-0.9}"
export CODE WORK VENV MODEL GPU_UTIL
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
fi

# run one of the python scripts in this folder: py gen_traces.py --out ...
py() { python "$CODE/$1" "${@:2}"; }
