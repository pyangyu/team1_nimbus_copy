#!/usr/bin/env bash
# Run the math agent through the Brownout gateway on AWS: bundle the commit, start the boxes, set up Open-Jev,
# deploy, sweep the policies, collect results into ./brownout-results (relative to the caller's directory).
# Usage: infra/brownout/e2e.sh [git-ref]    BROWNOUT_STOP=1 stops the boxes this run started.
set -euo pipefail
CALLER=$PWD
REF=${1:-HEAD}
source "$(dirname "$0")/env.sh"
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse "$REF")
TAG=${BROWNOUT_TAG:-$(date -u +%Y%m%d-%H%M%S)}
RESULTS=$CALLER/brownout-results
export BROWNOUT_STARTED_FILE=${BROWNOUT_STARTED_FILE:-/tmp/brownout-started-$$}
: > "$BROWNOUT_STARTED_FILE"

cleanup() {
  local started
  started=$(tr '\n' ' ' < "$BROWNOUT_STARTED_FILE" 2>/dev/null || true)
  if [[ "${BROWNOUT_STOP:-0}" == 1 && -n "${started// /}" ]]; then
    # shellcheck disable=SC2086
    ./down.sh $started || true
  fi
}
trap cleanup EXIT INT TERM

git -C "$ROOT" archive --format=tar.gz -o "/tmp/brownout-$SHA.tgz" "$SHA"
$AWS s3 cp --quiet "/tmp/brownout-$SHA.tgz" "s3://$BUCKET/ci/$SHA/src.tgz"

UP=$(./up.sh)
eval "$UP"

ssm_script "$BROWNOUT_JEV_ID" 3600 remote/setup_jev.sh
ssm_script "$BROWNOUT_JEV_ID" 1200 remote/deploy.sh "$SHA" jev "$BROWNOUT_LLM_URL"
ssm_script "$BROWNOUT_AGENT_ID" 1200 remote/deploy.sh "$SHA" agent
rc=0
ssm_script "$BROWNOUT_AGENT_ID" 12000 remote/run_e2e.sh "$SHA" "$BROWNOUT_GATEWAY_URL" "$TAG" || rc=$?
ssm_script "$BROWNOUT_JEV_ID" 600 remote/collect_jev.sh "$SHA" "$TAG" || true

mkdir -p "$RESULTS"
$AWS s3 cp --quiet --recursive "s3://$BUCKET/ci/$SHA/results/$TAG/" "$RESULTS/" || true
echo "results in $RESULTS (s3://$BUCKET/ci/$SHA/results/$TAG/)" >&2
exit $rc
