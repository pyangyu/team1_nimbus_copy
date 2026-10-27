#!/usr/bin/env bash
# Budget-drop baselines with the untrained model. Needs one GPU (we used an RTX 4090).
#   bash scripts/baselines.sh math         MATH-500 levels 4-5: natural traces, then the drop test
#   bash scripts/baselines.sh aime         AIME 2026, 8 samples per problem, full answers allowed
#   bash scripts/baselines.sh aime_final   same AIME traces, only the final answer may follow </think>
#   bash scripts/baselines.sh all
# aime and aime_final are also what the fine-tuned models are compared against.
# shellcheck source=/dev/null
source "$(dirname "$0")/env.sh"
step=${1:-all}
[ $# -gt 0 ] && shift

run_math() {
  py thinking_budget_bench.py --model "$MODEL" --levels 4 5 --n 200 --gpu-util "$GPU_UTIL" --out "$WORK/math_hard" \
    && py budget_drop_bench.py --model "$MODEL" --traces "$WORK/math_hard" --gpu-util "$GPU_UTIL" --out "$WORK/math_drop"
}
run_aime() {
  py aime_drop_bench.py --model "$MODEL" --gpu-util "$GPU_UTIL" --out "$WORK/aime_drop" "$@"
}
run_aime_final() {
  py aime_drop_bench.py --model "$MODEL" --gpu-util "$GPU_UTIL" --traces "$WORK/aime_drop" --out "$WORK/aime_final" \
    --answer-mode final "$@"
}

case "$step" in
  math)       run_math ;;
  aime)       run_aime "$@" ;;
  aime_final) run_aime_final "$@" ;;
  all)        run_math && run_aime && run_aime_final ;;
  *)          echo "unknown step $step (math | aime | aime_final | all)"; exit 1 ;;
esac
