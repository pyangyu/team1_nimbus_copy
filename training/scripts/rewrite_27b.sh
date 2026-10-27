#!/usr/bin/env bash
# Runs on a big GPU (we used one H100 80 GB on Lambda). Copy this training folder and
# $WORK/sft/rewrite_input.jsonl (made by make_sft.py) to that machine, then:
#   nohup bash scripts/rewrite_27b.sh rewrite_input.jsonl > rewrite_log.txt 2>&1 &
#   bash scripts/rewrite_27b.sh rewrite_input.jsonl Qwen/Some-Other-Model        to use another model
# It rewrites 20 examples first and prints two so you can check them (no headings, no \boxed, shorter than
# the budget), then does the rest. Output: rewrite_output.jsonl next to the input. Bring it back and run
# scripts/sft_rewrite.sh.
set -e
inp=${1:?usage: rewrite_27b.sh rewrite_input.jsonl [model]}
big=${2:-Qwen/Qwen3.6-27B}
code="$(cd "$(dirname "$0")/.." && pwd)"
out="$(dirname "$inp")/rewrite_output.jsonl"
VENV="${VENV:-$HOME/venv}"

echo "== setup $(date) =="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$VENV"; }
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -c "import vllm" 2>/dev/null || pip install -q -U pip vllm

echo "== test on 20 examples $(date) =="
python "$code/rewrite.py" --model "$big" --fp8 --inp "$inp" --out "${out%.jsonl}_test.jsonl" --limit 20
python - "$inp" "${out%.jsonl}_test.jsonl" <<'PY'
import json, sys
inp = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
rows = [json.loads(l) for l in open(sys.argv[2])]
for r in rows[:2]:
    print(f"\n--- {r['id']}  budget {inp[r['id']]['budget_tokens']} tokens, rewrite {r['out_tokens']} tokens")
    print(r["rewrite"][:1500])
print(f"\n{sum(r['finish'] == 'stop' for r in rows)}/{len(rows)} finished normally")
PY

echo "== all examples $(date) =="
python "$code/rewrite.py" --model "$big" --fp8 --inp "$inp" --out "$out"
echo "== done $(date): $(wc -l < "$out") rewrites in $out =="
