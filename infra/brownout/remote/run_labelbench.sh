#!/usr/bin/env bash
set -euo pipefail
SHA=$1 LLM=$2 TAG=$3
need_tools
OUT=/opt/brownout/results/$TAG/labelbench
mkdir -p "$OUT" /opt/brownout/data
cd /opt/brownout/src/agent
export LIFELINE_DATA=/opt/brownout/data
LIFELINE_API_KEY=$(api_key)
export LIFELINE_API_KEY
set +e
# shellcheck disable=SC2086
/opt/brownout/venv/bin/python -m math_agent.labelbench --base-url "$LLM" --out "$OUT" ${LABELBENCH_ARGS:-} > "$OUT/labelbench.log" 2>&1
rc=$?
set -e
aws s3 cp --region "$REGION" --quiet --recursive "$OUT" "s3://$BUCKET/ci/$SHA/results/$TAG/labelbench/"
tail -n 60 "$OUT/labelbench.log"
exit $rc
