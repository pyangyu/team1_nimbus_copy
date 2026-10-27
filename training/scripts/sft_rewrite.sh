#!/usr/bin/env bash
# SFT where the targets are 27B rewrites (compressed continuations), then the AIME test.
# Needs $WORK/sft from scripts/sft.sh (its samples are reused) and the output of scripts/rewrite_27b.sh.
# --extra-plain 6 adds a few more no-notice samples first, same as the run in results/sft_rewrite.
#   REWRITES=/path/to/rewrite_output.jsonl bash scripts/sft_rewrite.sh
# shellcheck source=/dev/null
source "$(dirname "$0")/env.sh"
: "${REWRITES:?set REWRITES to the rewrite_output.jsonl from rewrite_27b.sh}"

py make_sft.py --model "$MODEL" --gpu-util "$GPU_UTIL" --traces "$WORK/traces" --out "$WORK/sft_rewrite" \
     --reuse "$WORK/sft" --extra-plain 6 --rewrites "$REWRITES" --prefer-rewrite \
  && py train_lora.py --model "$MODEL" --data "$WORK/sft_rewrite/sft.jsonl" --out "$WORK/lora_rewrite" --merge --resume \
  && py aime_drop_bench.py --model "$WORK/lora_rewrite/merged" --gpu-util "$GPU_UTIL" --traces "$WORK/aime_drop" \
       --out "$WORK/aime_rewrite" --answer-mode final \
  && py compare_tuned.py --base "$WORK/aime_final" --tuned "$WORK/aime_rewrite"
