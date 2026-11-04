#!/usr/bin/env bash
# Runs on the Jev box: the settled-detector bench against vLLM (private IP) and the local Open-Jev servers.
set -euo pipefail
SHA=$1 LLM=$2 TAG=$3
need_tools
OUT=/opt/brownout/results/$TAG/settledbench
mkdir -p "$OUT" /opt/brownout/data
cd /opt/brownout/src/agent
export LIFELINE_DATA=/opt/brownout/data
LIFELINE_API_KEY=$(api_key)
export LIFELINE_API_KEY
JEV_URLS=""
for port in $JEV_PORTS; do JEV_URLS+="${JEV_URLS:+,}http://127.0.0.1:$port/v1/systemone"; done
set +e
# shellcheck disable=SC2086
/opt/brownout/venv/bin/python -m math_agent.settledbench --base-url "$LLM" --jev-url "$JEV_URLS" --out "$OUT"   ${SETTLEDBENCH_ARGS:-} > "$OUT/settledbench.log" 2>&1
rc=$?
set -e
aws s3 cp --region "$REGION" --quiet --recursive "$OUT" "s3://$BUCKET/ci/$SHA/results/$TAG/settledbench/"
tail -n 80 "$OUT/settledbench.log"
exit $rc
