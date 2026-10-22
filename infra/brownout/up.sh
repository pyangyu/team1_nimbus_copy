#!/usr/bin/env bash
# Start the three Brownout boxes and wait until vLLM, Open-Jev and SSM are ready.
# Prints BROWNOUT_* variables on stdout (eval-able); BROWNOUT_STARTED lists the boxes this call started.
set -euo pipefail
source "$(dirname "$0")/env.sh"

IDS=() STARTED=()
for name in "$JEV_NAME" "$LLM_NAME" "$AGENT_NAME"; do
  id=$(instance_id "$name")
  [[ "$id" != "None" ]] || { echo "no $name instance: run setup_boxes.sh" >&2; exit 1; }
  state=$($AWS ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$state" == stopping ]]; then $AWS ec2 wait instance-stopped --instance-ids "$id"; state=stopped; fi
  if [[ "$state" == stopped ]]; then
    for try in 1 2 3 4 5 6; do
      if $AWS ec2 start-instances --instance-ids "$id" >/dev/null 2>/tmp/brownout-start.err; then break; fi
      grep -q InsufficientInstanceCapacity /tmp/brownout-start.err || { cat /tmp/brownout-start.err >&2; exit 1; }
      [[ $try -lt 6 ]] || { echo "no capacity for $name after $try tries" >&2; exit 1; }
      echo "no capacity for $name in its zone, retrying in 60 s ($try/6)" >&2
      sleep 60
    done
    STARTED+=("$id")
    [[ -z "${BROWNOUT_STARTED_FILE:-}" ]] || echo "$id" >> "$BROWNOUT_STARTED_FILE"
    echo "started $name $id" >&2
  fi
  IDS+=("$id")
done
JEV=${IDS[0]} LLM=${IDS[1]} AGENT=${IDS[2]}
echo "BROWNOUT_STARTED=\"${STARTED[*]:-}\""
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

ip() { $AWS ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text; }
LLM_IP=$(ip "$LLM") JEV_IP=$(ip "$JEV")
echo "BROWNOUT_JEV_ID=$JEV"
echo "BROWNOUT_LLM_ID=$LLM"
echo "BROWNOUT_AGENT_ID=$AGENT"
echo "BROWNOUT_LLM_URL=http://$LLM_IP:8000/v1"
echo "BROWNOUT_GATEWAY_URL=http://$JEV_IP:9000/v1"
ssm_script "$JEV" 1800 remote/wait_ready.sh "http://$LLM_IP:8000/v1"
