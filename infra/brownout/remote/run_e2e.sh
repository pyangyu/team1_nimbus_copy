#!/usr/bin/env bash
set -euo pipefail
SHA=$1 GATEWAY=$2 TAG=$3 HARD=${4-math500}
need_tools
OUT=/opt/brownout/results/$TAG
mkdir -p "$OUT" /opt/brownout/data
cd /opt/brownout/src/agent
export LIFELINE_BASE_URL=$GATEWAY LIFELINE_DATA=/opt/brownout/data MATH_AGENT_LIVE=1 BROWNOUT_LIVE=1 BROWNOUT_OUT=$OUT BROWNOUT_HARD_DATASET=$HARD
LIFELINE_API_KEY=$(api_key)
export LIFELINE_API_KEY
set +e
/opt/brownout/venv/bin/python -m pytest -q -rs tests/test_live.py tests/test_brownout_live.py \
  --junitxml="$OUT/junit.xml" > "$OUT/pytest.log" 2>&1
rc=$?
set -e
aws s3 cp --region "$REGION" --quiet --recursive "$OUT" "s3://$BUCKET/ci/$SHA/results/$TAG/"
tail -n 40 "$OUT/pytest.log"
[[ ! -f "$OUT/results.md" ]] || cat "$OUT/results.md"
exit $rc
