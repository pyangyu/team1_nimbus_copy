#!/usr/bin/env bash
# Cut-aware SFT on Qwen3-4B with LoRA, then the AIME before/after test.
# One 24 GB GPU (we used an RTX 4090). Each step takes hours.
#   bash scripts/sft.sh                  all four steps
#   bash scripts/sft.sh traces|data|train|eval
# Each step picks up where it stopped, so after a crash just run the same command again.
# The eval step needs $WORK/aime_drop and $WORK/aime_final from scripts/baselines.sh.
# shellcheck source=/dev/null
source "$(dirname "$0")/env.sh"
step=${1:-all}
[ $# -gt 0 ] && shift

run_traces() { py gen_traces.py --model "$MODEL" --gpu-util "$GPU_UTIL" --out "$WORK/traces" "$@"; }
run_data()   { py make_sft.py --model "$MODEL" --gpu-util "$GPU_UTIL" --traces "$WORK/traces" --out "$WORK/sft" "$@"; }
run_train()  { py train_lora.py --model "$MODEL" --data "$WORK/sft/sft.jsonl" --out "$WORK/lora_sft" --merge --resume "$@"; }
run_eval() {
  [ -f "$WORK/aime_drop/phase_a.jsonl" ] || { echo "run scripts/baselines.sh aime and aime_final first"; return 1; }
  py aime_drop_bench.py --model "$WORK/lora_sft/merged" --gpu-util "$GPU_UTIL" --traces "$WORK/aime_drop" \
       --out "$WORK/aime_sft" --answer-mode final "$@" \
    && py compare_tuned.py --base "$WORK/aime_final" --tuned "$WORK/aime_sft"
}

case "$step" in
  traces) run_traces "$@" ;;
  data)   run_data "$@" ;;
  train)  run_train "$@" ;;
  eval)   run_eval "$@" ;;
  all)    run_traces && run_data && run_train && run_eval ;;
  *)      echo "unknown step $step (traces | data | train | eval | all)"; exit 1 ;;
esac
