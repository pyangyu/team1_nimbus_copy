#!/usr/bin/env bash
# Run the math agent through the Brownout gateway on AWS: bundle the commit, start the boxes, set up Open-Jev,
# deploy, sweep the policies, collect results into ./brownout-results (relative to the caller's directory).
# Usage: infra/brownout/e2e.sh [git-ref]    BROWNOUT_STOP=1 stops the boxes this run started.
# BROWNOUT_HARD_DATASET= (empty) runs the GSM8K sweep only. BROWNOUT_SUITE=realistic runs the capacity + rate-profile suite
# (BROWNOUT_TRACE=burstgpt replays BurstGPT arrivals instead of the profile),
# without Jev, with a second vLLM replica on the Jev box's GPU (BROWNOUT_LLM_ON_JEV=0 keeps one). Both replicas are
# (re)set to config.env's model, or BROWNOUT_MODEL, plus BROWNOUT_LORA=name=s3-key (a LoRA tarball under ci/).
set -euo pipefail
CALLER=$PWD
REF=${1:-HEAD}
source "$(dirname "$0")/env.sh"
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse "$REF")
TAG=${BROWNOUT_TAG:-$(date -u +%Y%m%d-%H%M%S)}
RESULTS=$CALLER/brownout-results
export BROWNOUT_STARTED_FILE=${BROWNOUT_STARTED_FILE:-/tmp/brownout-started-$$}
touch "$BROWNOUT_STARTED_FILE"

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

UPSTREAMS=$BROWNOUT_LLM_URL
if [[ "${BROWNOUT_SUITE:-default}" == realistic ]]; then
  export BROWNOUT_JEV=${BROWNOUT_JEV:-0}
  BROWNOUT_LLM_ON_JEV=${BROWNOUT_LLM_ON_JEV:-1}
fi
if [[ "${BROWNOUT_LLM_ON_JEV:-0}" == 1 ]]; then
  [[ "${BROWNOUT_JEV:-1}" == 0 ]] || { echo "BROWNOUT_LLM_ON_JEV=1 needs BROWNOUT_JEV=0 (one GPU on the Jev box)" >&2; exit 1; }
  cfg() { grep -E "^$1=" ../vllm-ec2/config.env | cut -d= -f2-; }
  model=${BROWNOUT_MODEL:-$(cfg MODEL)}
  for id in "$BROWNOUT_LLM_ID" "$BROWNOUT_JEV_ID"; do
    ssm_script "$id" 3600 remote/setup_llm.sh "$(cfg VLLM_IMAGE)" "$model" "$(cfg MAX_MODEL_LEN)" "${BROWNOUT_LORA:-}"
  done
  ssm_script "$BROWNOUT_LLM_ID" 300 remote/show_vllm.sh || true
  UPSTREAMS="http://127.0.0.1:8000/v1,$BROWNOUT_LLM_URL"
fi
[[ "${BROWNOUT_JEV:-1}" == 0 ]] || ssm_script "$BROWNOUT_JEV_ID" 3600 remote/setup_jev.sh
ssm_script "$BROWNOUT_JEV_ID" 1200 remote/deploy.sh "$SHA" jev "$UPSTREAMS"
ssm_script "$BROWNOUT_AGENT_ID" 1200 remote/deploy.sh "$SHA" agent
AGENT_TIMEOUT=12000
if [[ "${BROWNOUT_SUITE:-default}" == realistic ]]; then
  profile_s=0
  IFS=, read -ra segs <<< "${BROWNOUT_PROFILE:-180:0.7,120:1.2,180:0.7,180:1.5,240:0.7,120:2.0,180:0.7,240:1.2,300:0.7}"
  for s in "${segs[@]}"; do d=${s%%:*}; profile_s=$((profile_s + ${d%.*})); done
  [[ -z "${BROWNOUT_TRACE:-}" ]] || { t=${BROWNOUT_TRACE_S:-1800}; profile_s=${t%.*}; }
  IFS=, read -ra pols <<< "${BROWNOUT_POLICIES:-none,uniform,brownout:silent:online,brownout:announced:online}"
  tail_s=${BROWNOUT_TAIL_S:-1200}
  warm_s=${BROWNOUT_CAPACITY_WARMUP_S:-180} meas_s=${BROWNOUT_CAPACITY_S:-600}
  cap_s=$(( ${warm_s%.*} + ${meas_s%.*} + 1800 ))
  [[ -z "${BROWNOUT_CAPACITY:-}" ]] || cap_s=900
  AGENT_TIMEOUT=$(( cap_s + ${#pols[@]} * (profile_s + ${tail_s%.*} + 1800) + 1800 ))
fi
AGENT_TIMEOUT=${BROWNOUT_AGENT_TIMEOUT:-$AGENT_TIMEOUT}
echo "agent box timeout ${AGENT_TIMEOUT}s" >&2
rc=0
ssm_script "$BROWNOUT_AGENT_ID" "$AGENT_TIMEOUT" remote/run_e2e.sh "$SHA" "$BROWNOUT_GATEWAY_URL" "$TAG" "${BROWNOUT_HARD_DATASET-math500}" || rc=$?
ssm_script "$BROWNOUT_JEV_ID" 600 remote/collect_jev.sh "$SHA" "$TAG" || true

mkdir -p "$RESULTS"
$AWS s3 cp --quiet --recursive "s3://$BUCKET/ci/$SHA/results/$TAG/" "$RESULTS/" || true
echo "results in $RESULTS (s3://$BUCKET/ci/$SHA/results/$TAG/)" >&2
exit $rc
