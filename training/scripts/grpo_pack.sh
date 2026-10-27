#!/usr/bin/env bash
# Run where the SFT results live. Packs everything the GRPO machine needs into $WORK/grpo_pack.zip:
# GRPO prompts (MATH train cut points), the SFT adapter, the AIME traces, the untrained model's AIME results
# for the comparison, and the scripts. Then copy the zip to an 80 GB GPU and follow scripts/grpo_run.sh.
#   bash scripts/grpo_pack.sh
# shellcheck source=/dev/null
source "$(dirname "$0")/env.sh"
set -e
pack="$WORK/grpo_pack"
rm -rf "$pack" && mkdir -p "$pack/aime_drop" "$pack/aime_final"
py prep_grpo.py --model "$MODEL" --traces "$WORK/traces" --examples "$WORK/sft/examples.jsonl" \
   --out "$pack/grpo_prompts.jsonl"
cp -r "$WORK/lora_sft/adapter" "$pack/init_adapter"
cp "$WORK/aime_drop/phase_a.jsonl" "$pack/aime_drop/"
cp "$WORK/aime_final/jobs.jsonl" "$WORK/aime_final/config.json" "$pack/aime_final/"
cp "$CODE"/*.py "$CODE/scripts/grpo_run.sh" "$pack/"
rm -f "$WORK/grpo_pack.zip"
(cd "$WORK" && python -m zipfile -c grpo_pack.zip grpo_pack)
ls -la "$WORK/grpo_pack.zip"
