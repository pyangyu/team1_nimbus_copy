#!/usr/bin/env bash
# Serving experiments for Table II, the load sweep and fig_reaction.
# vLLM must already be running (see the comment at the bottom). Usage:
#   bash run_serve.sh calibrate          # 1. measure capacity (req/s) with no cut at low load
#   CAP_RPS=0.40 bash run_serve.sh main  # 2. Table II: all policies, base 80% + two 4.5x bursts (mean ~125%)
#   CAP_RPS=0.40 bash run_serve.sh sweep # 3. sweep table: goodput vs mean load
#   CAP_RPS=0.40 bash run_serve.sh select # 4. selection table + ablation, 120% load
set -euo pipefail
cd "$(dirname "$0")"

TRACES=${TRACES:-traces/Qwen__Qwen3.8-27B.eval.jsonl}   # problems to serve (prompt + gold)
MODEL=${MODEL:-Qwen/Qwen3.8-27B}                       # the served-model-name in vLLM
ADAPTER=${ADAPTER:-cut}                                # --lora-modules name of the cut adapter
URL=${URL:-http://localhost:8000}
SLOTS=${SLOTS:-32}
SLO=${SLO:-10}
DUR=${DUR:-120}                                        # seconds of arrivals
OUT=${OUT:-serve}
mkdir -p "$OUT"

run() {   # run <policy> <rate> <bursts> <tag>
  python serve_eval.py "$TRACES" --policy "$1" --rate "$2" --bursts "$3" --duration "$DUR" \
    --slots "$SLOTS" --slo "$SLO" --model "$MODEL" --adapter "$ADAPTER" --base-url "$URL" \
    --out "$OUT/$4.$1.jsonl" 2>&1 | tee "$OUT/$4.$1.log" | grep -E "arrivals|done ->"
}

case "${1:-}" in
calibrate)
  # low load, no bursts: requests rarely wait, so service time = finish - admit is clean
  run nocut 0.05 "" calib
  python serve_report.py "$OUT/calib.nocut.jsonl" | tail -1
  echo "Use about 0.8 x this as the slot-limited capacity if the batch was not full; then set CAP_RPS."
  ;;
main)
  : "${CAP_RPS:?set CAP_RPS from the calibrate step}"
  RATE=$(python -c "print(round(0.8 * $CAP_RPS, 4))")
  BURSTS=${BURSTS:-"20:10:4.5,70:10:4.5"}             # with DUR=120: mean load 0.8*(1+2*10*3.5/120) = 1.27
  for p in nocut reject budget exit hard tb; do run $p "$RATE" "$BURSTS" main; done
  python serve_report.py "$OUT"/main.*.jsonl --ref "$OUT/main.nocut.jsonl" --timeline 5 | tee "$OUT/main.txt"
  ;;
sweep)
  : "${CAP_RPS:?set CAP_RPS from the calibrate step}"
  for L in 0.8 1.0 1.2 1.4 1.6; do
    RATE=$(python -c "print(round($L * $CAP_RPS, 4))")
    for p in nocut reject budget tb; do run $p "$RATE" "" "load$L"; done
    python serve_report.py "$OUT"/load$L.*.jsonl --ref "$OUT/load$L.nocut.jsonl" | tee "$OUT/load$L.txt"
  done
  ;;
select)
  : "${CAP_RPS:?set CAP_RPS from the calibrate step}"
  RATE=$(python -c "print(round(1.2 * $CAP_RPS, 4))")
  run nocut "$RATE" "" sel
  for s in random written low high oracle; do                       # tab:alloc (hard stop, so only selection differs)
    python serve_eval.py "$TRACES" --policy hard --select $s --rate "$RATE" --bursts "" --duration "$DUR" \
      --slots "$SLOTS" --slo "$SLO" --model "$MODEL" --base-url "$URL" --out "$OUT/sel.hard_$s.jsonl" > "$OUT/sel.hard_$s.log" 2>&1
  done
  # tab:ablation: hard_random -> hard_high -> SFT adapter -> SFT+GRPO adapter
  ADAPTER=${ADAPTER_SFT:-cut-sft} run tb "$RATE" "" sel_sft
  ADAPTER=${ADAPTER:-cut} run tb "$RATE" "" sel_grpo
  python serve_report.py "$OUT"/sel.*.jsonl "$OUT"/sel_*.tb.jsonl --ref "$OUT/sel.nocut.jsonl" | tee "$OUT/select.txt"
  ;;
*)
  echo "usage: bash run_serve.sh calibrate | main | sweep | select"; exit 1 ;;
esac

# vLLM, started once in another shell (same env as before; no -U installs):
#   cd ~/thinkingcut && PYTHONPATH=$PWD TB_ALORA=cut,cut-sft TB_KAPPA=2048 \
#   vllm serve Qwen/Qwen3.8-27B --reasoning-parser qwen3 --max-num-seqs 64 --no-async-scheduling \
#       --enable-lora --max-lora-rank 16 --max-loras 2 \
#       --lora-modules cut=adapters/cut-grpo cut-sft=adapters/cut-sft \
#       --logits-processors tb_vllm:TBCut
# Same server for every policy: the cuts happen inside it (tb_vllm.py), in place. max-num-seqs above SLOTS
# leaves room for the rating calls and the exit probes. Run tb_vllm_check.py once before the experiments.
