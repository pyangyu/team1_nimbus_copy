#!/usr/bin/env bash
# Round 2 on one 80 GB GPU (we use an H100 on Lambda). Two parts:
#   grpo  harder GRPO, starting from the first GRPO adapter: natural traces on competition problems
#         (DeepScaleR), drop the ones it never solves, cut the rest, train 100 steps
#   eval  AIME 2026 plus HMMT Feb 2026, HMMT Nov 2025, BRUMO 2025 and SMT 2025, each for the untrained
#         model, round 1 and round 2, all on this GPU, then paired comparisons
# Put in this folder: r1_adapter/ (adapter_config.json + adapter_model.safetensors) and
# aime_drop/phase_a.jsonl (the AIME natural traces every earlier run used). Then:
#   nohup bash round2_run.sh > log.txt 2>&1 &          then: tail -f log.txt
#   nohup bash round2_run.sh eval > log.txt 2>&1 &     one part only (check | grpo | eval | pack)
# Every step skips what is already done, so after a stop just start it again.
# At the end: results_round2.tgz (small: summaries, comparisons, logs) and r2_adapter.tgz (the new model).
cd "$(dirname "$0")" || exit 1
step=${1:-all}
VENV="${VENV:-$HOME/venv}"
SETS=(hmmt_feb_2026 hmmt_nov_2025 brumo_2025 smt_2025)
HARD=agentica-org/DeepScaleR-Preview-Dataset
EVAL=(--answer-mode final --max-num-seqs 64)

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

check() {
  echo "== check inputs and datasets $(date) =="
  [ -f r1_adapter/adapter_config.json ] && [ -f r1_adapter/adapter_model.safetensors ] \
    || { echo "!! r1_adapter/ needs adapter_config.json and adapter_model.safetensors"; return 1; }
  [ -f aime_drop/phase_a.jsonl ] || { echo "!! aime_drop/phase_a.jsonl not found"; return 1; }
  python - "$HARD" "${SETS[@]}" <<'PY'
import sys
from datasets import load_dataset
print(sys.argv[1], len(load_dataset(sys.argv[1], split="train")), "problems")
for s in ["aime_2026", *sys.argv[2:]]:
    print(f"MathArena/{s}", len(load_dataset(f"MathArena/{s}", split="train")), "problems")
PY
}

grpo() {
  echo "== hard traces $(date) =="
  [ -s hard/summary.json ] || python gen_traces.py --dataset "$HARD" --n 1600 --samples 4 --max-think 14000 \
      --max-num-seqs 128 --chunk 2048 --out hard || return 1
  echo "== GRPO prompts $(date) =="
  [ -s hard_prompts.jsonl ] || python prep_grpo.py --traces hard --out hard_prompts.jsonl \
      --pass-range 0.25 1.0 --cuts 3 || return 1
  echo "== GRPO round 2 $(date) =="
  [ -d r2/merged ] || python grpo.py --init-adapter r1_adapter --prompts hard_prompts.jsonl --out r2 \
      --no-seen --groups 8 --oversample 2 --steps 100 --max-len 10240 --resume || return 1
}

# one eval run: set, name, model, natural traces to reuse (empty = make them)
run_one() {
  local set=$1 name=$2 model=$3 traces=$4 out=evals/$1/$2
  [ -s "$out/summary.json" ] && return 0
  echo "== eval $set $name $(date) =="
  python aime_drop_bench.py --model "$model" --dataset "MathArena/$set" --out "$out" --resume \
    ${traces:+--traces "$traces"} "${EVAL[@]}"
}

compare() {   # set, base name, tuned name
  local d=evals/$1
  [ -s "$d/$2/jobs.jsonl" ] && [ -s "$d/$3/jobs.jsonl" ] || return 0
  python compare_tuned.py --base "$d/$2" --tuned "$d/$3" --save "$d/$3_vs_$2.json" > "$d/$3_vs_$2.txt"
}

evals() {
  [ -d r1_merged ] || python merge_lora.py --adapter r1_adapter --out r1_merged || return 1
  local models=(base:Qwen/Qwen3-4B r1:r1_merged)
  [ -d r2/merged ] && models+=(r2:r2/merged)
  for set in aime_2026 "${SETS[@]}"; do
    local base_traces=""
    [ "$set" = aime_2026 ] && base_traces=aime_drop
    run_one "$set" base Qwen/Qwen3-4B "$base_traces" || continue
    for m in "${models[@]:1}"; do
      run_one "$set" "${m%%:*}" "${m#*:}" "evals/$set/base"
    done
    compare "$set" base r1
    compare "$set" base r2
    compare "$set" r1 r2
  done
}

pack() {
  echo "== pack $(date) =="
  tar czf results_round2.tgz --ignore-failed-read hard/summary.json r2/log.jsonl log.txt \
    evals/*/*.txt evals/*/*.json evals/*/*/summary.json evals/*/*/config.json 2>/dev/null
  [ -d r2/adapter ] && tar czf r2_adapter.tgz -C r2 adapter
  ls -la results_round2.tgz r2_adapter.tgz 2>/dev/null
}

case "$step" in
  check) check ;;
  grpo)  check && grpo ;;
  eval)  check && evals; pack ;;
  pack)  pack ;;
  all)   check || exit 1
         grpo || echo "!! GRPO stopped early; running the evals without round 2"
         evals
         pack ;;
  *)     echo "unknown step $step (check | grpo | eval | pack | all)"; exit 1 ;;
esac
echo "== done $(date) =="
