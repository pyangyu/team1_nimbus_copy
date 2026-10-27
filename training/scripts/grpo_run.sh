#!/usr/bin/env bash
# GRPO on top of the SFT adapter, then AIME evals of both models on the same GPU. Needs one 80 GB GPU
# (we used an H100 on Nebius; the 40 training steps took about 32 min, setup and the two evals come on top).
# On that machine:
#   python3 -m zipfile -e grpo_pack.zip . && cd grpo_pack
#   nohup bash grpo_run.sh > log.txt 2>&1 &        then: tail -f log.txt
#   nohup bash grpo_run.sh --resume --steps 40 > log.txt 2>&1 &     finish from the last checkpoint
# Extra arguments go to grpo.py and win over the defaults below. Results: compare_grpo_vs_base.txt
# (against the untrained model's silent cut) and compare_grpo_vs_sft.txt (against the SFT model, same GPU).
cd "$(dirname "$0")" || exit 1
VENV="${VENV:-$HOME/venv}"
echo "== setup $(date) =="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$VENV"; }
fi
# Triton compiles small C helpers at start-up and needs the Python headers
pyver=$(python3 -c 'import sys; print(sys.version_info[1])')
[ -f "/usr/include/python3.$pyver/Python.h" ] || \
  { sudo apt-get update -qq && sudo apt-get install -y -qq python3-dev build-essential; }
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -c "import vllm, peft, datasets, math_verify" 2>/dev/null || pip install -q -U pip vllm peft datasets math-verify
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv

# the settings behind our result: half the budget, only prompts without the answer yet, 40 steps
# (accuracy stopped rising after ~20 steps while the model kept closing earlier)
echo "== GRPO $(date) =="
python grpo.py --init-adapter init_adapter --prompts grpo_prompts.jsonl --out grpo_out \
  --w-scale 0.5 --no-seen --oversample 3 --groups 12 --steps 40 "$@" || exit 1

echo "== merge the SFT adapter for the control eval $(date) =="
[ -d sft_merged ] || python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "init_adapter").merge_and_unload()
m.save_pretrained("sft_merged", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B").save_pretrained("sft_merged")
PY

echo "== AIME eval: SFT model (control, same GPU) $(date) =="
python aime_drop_bench.py --model sft_merged --traces aime_drop --out aime_sft_h100 --answer-mode final --max-num-seqs 64
echo "== AIME eval: GRPO model $(date) =="
python aime_drop_bench.py --model grpo_out/merged --traces aime_drop --out aime_grpo --answer-mode final --max-num-seqs 64

python compare_tuned.py --base aime_final --tuned aime_grpo | tee compare_grpo_vs_base.txt
python compare_tuned.py --base aime_sft_h100 --tuned aime_grpo | tee compare_grpo_vs_sft.txt
echo "== done $(date) =="
