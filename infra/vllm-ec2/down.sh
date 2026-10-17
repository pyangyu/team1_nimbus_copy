#!/usr/bin/env bash
# Stop the vLLM instance (EBS and the cached model are kept, ~$10/month).
# ./down.sh --terminate deletes it entirely.
set -euo pipefail
export AWS_PAGER=""
AWS="aws --region ${AWS_REGION:-us-east-1}"
ID=$($AWS ec2 describe-instances --filters Name=tag:Name,Values=lifeline-vllm \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [[ "$ID" == "None" ]]; then echo "no lifeline-vllm instance"; exit 0; fi
if [[ "${1:-}" == "--terminate" ]]; then
  $AWS ec2 terminate-instances --instance-ids "$ID" >/dev/null
  $AWS ec2 wait instance-terminated --instance-ids "$ID"
  echo "terminated $ID"
else
  $AWS ec2 stop-instances --instance-ids "$ID" >/dev/null
  $AWS ec2 wait instance-stopped --instance-ids "$ID"
  echo "stopped $ID"
fi
