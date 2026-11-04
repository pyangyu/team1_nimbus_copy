#!/usr/bin/env bash
# Run the label fidelity bench (agent/math_agent/labelbench.py) on AWS: only the vLLM box and the agent box.
# Usage: infra/brownout/labelbench.sh [git-ref]    BROWNOUT_STOP=1 stops the boxes this run started.
# LABELBENCH_ARGS passes extra flags to the bench (e.g. "--n 32 --seed 0 --budget 4096").
set -euo pipefail
CALLER=$PWD
REF=${1:-HEAD}
source "$(dirname "$0")/env.sh"
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse "$REF")
TAG=${BROWNOUT_TAG:-labelbench-$(date -u +%Y%m%d-%H%M%S)}
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

TMPD=$( (command -v cygpath >/dev/null && cygpath -m /tmp) || echo /tmp)
git -C "$ROOT" archive --format=tar.gz -o "$TMPD/brownout-$SHA.tgz" "$SHA"
$AWS s3 cp --quiet "$TMPD/brownout-$SHA.tgz" "s3://$BUCKET/ci/$SHA/src.tgz"

IDS=()
for name in "$LLM_NAME" "$AGENT_NAME"; do
  id=$(instance_id "$name")
  [[ "$id" != "None" ]] || { echo "no $name instance" >&2; exit 1; }
  state=$($AWS ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$state" == stopping ]]; then $AWS ec2 wait instance-stopped --instance-ids "$id"; state=stopped; fi
  if [[ "$state" == stopped ]]; then
    for try in 1 2 3 4 5 6 7 8 9 10; do
      if $AWS ec2 start-instances --instance-ids "$id" >/dev/null 2>/tmp/brownout-start.err; then break; fi
      grep -q InsufficientInstanceCapacity /tmp/brownout-start.err || { cat /tmp/brownout-start.err >&2; exit 1; }
      [[ $try -lt 10 ]] || { echo "no capacity for $name after $try tries" >&2; exit 1; }
      echo "no capacity for $name in its zone, retrying in 60 s ($try/10)" >&2
      sleep 60
    done
    echo "$id" >> "$BROWNOUT_STARTED_FILE"
    echo "started $name $id" >&2
  fi
  IDS+=("$id")
done
LLM=${IDS[0]} AGENT=${IDS[1]}
$AWS ec2 wait instance-running --instance-ids "${IDS[@]}"
for id in "${IDS[@]}"; do
  for _ in $(seq 1 60); do
    ping=$($AWS ssm describe-instance-information --filters "Key=InstanceIds,Values=$id" \
      --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
    [[ "$ping" == Online ]] && break
    sleep 10
  done
  [[ "$ping" == Online ]] || { echo "SSM agent on $id did not come online" >&2; exit 1; }
done
LLM_IP=$($AWS ec2 describe-instances --instance-ids "$LLM" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
LLM_URL=http://$LLM_IP:8000/v1
echo "vLLM at $LLM_URL, agent $AGENT" >&2

ssm_script "$AGENT" 1800 remote/wait_ready.sh "$LLM_URL"
ssm_script "$AGENT" 1200 remote/deploy.sh "$SHA" agent
rc=0
ssm_script "$AGENT" 7200 remote/run_labelbench.sh "$SHA" "$LLM_URL" "$TAG" || rc=$?
RESULTS=$( (command -v cygpath >/dev/null && cygpath -m "$CALLER") || echo "$CALLER")/brownout-results/$TAG
mkdir -p "$RESULTS"
$AWS s3 cp --quiet --recursive "s3://$BUCKET/ci/$SHA/results/$TAG/labelbench/" "$RESULTS/" || true
echo "results in $RESULTS (s3://$BUCKET/ci/$SHA/results/$TAG/labelbench/)" >&2
exit $rc
