#!/usr/bin/env bash
# Stop the vLLM instances (EBS and the cached model are kept, ~$10/month each).
# Usage: ./down.sh [batch] [--terminate]   (--terminate deletes them entirely)
set -euo pipefail
source "$(dirname "$0")/env.sh"

TERMINATE=0
TIERS=()
for a in "$@"; do
  if [[ "$a" == --terminate ]]; then TERMINATE=1; else TIERS+=("$a"); fi
done
[[ ${#TIERS[@]} -gt 0 ]] || TIERS=(batch)

for tier in "${TIERS[@]}"; do
  name=lifeline-vllm-$tier
  ID=$($AWS ec2 describe-instances --filters Name=tag:Name,Values=$name \
    Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)
  if [[ "$ID" == "None" ]]; then echo "no $name instance"; continue; fi
  if [[ $TERMINATE == 1 ]]; then
    $AWS ec2 terminate-instances --instance-ids "$ID" >/dev/null
    $AWS ec2 wait instance-terminated --instance-ids "$ID"
    echo "terminated $name $ID"
  else
    $AWS ec2 stop-instances --instance-ids "$ID" >/dev/null
    $AWS ec2 wait instance-stopped --instance-ids "$ID"
    echo "stopped $name $ID"
  fi
done
