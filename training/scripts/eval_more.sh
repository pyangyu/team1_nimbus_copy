#!/usr/bin/env bash
# Two extra checks for a fine-tuned model (default: the SFT one in $WORK/lora_sft).
#   leakage  base traces, cut as usual, but the answer may be long: does reasoning move into the answer?
#   own      the tuned model writes the whole trace itself (how it would run in serving), no-cut regression too
#   bash scripts/eval_more.sh [leakage|own|all] [adapter folder under $WORK]
# shellcheck source=/dev/null
source "$(dirname "$0")/env.sh"
step=${1:-all}
name=${2:-lora_sft}
merged="$WORK/$name/merged"
[ -d "$merged" ] || { echo "$merged not found"; exit 1; }

run_leakage() {
  py aime_drop_bench.py --model "$merged" --gpu-util "$GPU_UTIL" --traces "$WORK/aime_drop" \
       --out "$WORK/aime_${name}_full" --answer-mode full \
    && py compare_tuned.py --base "$WORK/aime_drop" --tuned "$WORK/aime_${name}_full"
}
run_own() {
  py aime_drop_bench.py --model "$merged" --gpu-util "$GPU_UTIL" --out "$WORK/aime_${name}_own" --answer-mode final --resume
}

case "$step" in
  leakage) run_leakage ;;
  own)     run_own ;;
  all)     run_leakage; run_own ;;
  *)       echo "unknown step $step (leakage | own | all)"; exit 1 ;;
esac
