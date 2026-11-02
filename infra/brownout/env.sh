#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export AWS_PAGER="" MSYS_NO_PATHCONV=1
AWS_REGION=${AWS_REGION:-us-east-1}
AWS="aws --region $AWS_REGION"
ACCOUNT=826671498662
BUCKET=capstone-ci-$ACCOUNT
JEV_NAME=brownout-jev
LLM_NAME=lifeline-vllm-batch
AGENT_NAME=brownout-agent
SG_NAME=brownout-internal-sg
PROFILE_NAME=brownout-ec2
export BUCKET JEV_NAME LLM_NAME AGENT_NAME SG_NAME PROFILE_NAME

if [[ "$($AWS sts get-caller-identity --query Account --output text)" != "$ACCOUNT" ]]; then
  echo "AWS credentials are not for account $ACCOUNT" >&2
  exit 1
fi

instance_id() {
  $AWS ec2 describe-instances --filters "Name=tag:Name,Values=$1" \
    Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[0].Instances[0].InstanceId' --output text
}

ssm_script() {
  local id=$1 timeout=$2 script=$3
  shift 3
  local b64 args="" envs="" cmd params cid st v
  b64=$(cat remote/common.sh "$script" | base64 | tr -d '\n')
  for a in "$@"; do args+=" $(printf '%q' "$a")"; done
  for v in JEV_PORTS JEV_CONCURRENCY JEV_TIMEOUT BROWNOUT_POLICIES BROWNOUT_N BROWNOUT_SKIP_AGENT_LIVE BROWNOUT_SUITE \
    BROWNOUT_WORKLOAD BROWNOUT_PROFILE BROWNOUT_CAPACITY BROWNOUT_CAPACITY_S BROWNOUT_CAPACITY_WARMUP_S \
    BROWNOUT_CAPACITY_CONCURRENCY BROWNOUT_TAIL_S BROWNOUT_SEED BROWNOUT_JEV BROWNOUT_TRACE BROWNOUT_TRACE_LOG \
    BROWNOUT_TRACE_START_H BROWNOUT_TRACE_WINDOW_H BROWNOUT_TRACE_S BROWNOUT_LOAD BROWNOUT_NOTICE BROWNOUT_MODEL \
    BROWNOUT_WARMUP_N; do
    [[ -z "${!v:-}" ]] || envs+="$v=$(printf '%q' "${!v}") "
  done
  cmd="echo $b64 | base64 -d > /tmp/ssm-script.sh && ${envs}bash /tmp/ssm-script.sh$args"
  params=$(python3 -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": [sys.argv[2]]}))' "$cmd" "$timeout")
  cid=$($AWS ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
    --parameters "$params" --timeout-seconds 600 --query Command.CommandId --output text)
  echo "ssm $(basename "$script") on $id: command $cid" >&2
  while true; do
    st=$($AWS ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query Status --output text 2>/dev/null || echo Pending)
    case "$st" in
      Pending|InProgress|Delayed) sleep 10 ;;
      *) break ;;
    esac
  done
  $AWS ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query StandardOutputContent --output text >&2
  $AWS ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query StandardErrorContent --output text | tail -n 40 >&2
  echo "ssm $(basename "$script") on $id: $st" >&2
  [[ "$st" == Success ]]
}
