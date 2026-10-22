#!/usr/bin/env bash
# Stop Brownout boxes. Usage: ./down.sh [instance-id ...]   (default: all three)
set -euo pipefail
source "$(dirname "$0")/env.sh"

IDS=("$@")
if [[ ${#IDS[@]} -eq 0 ]]; then
  for name in "$JEV_NAME" "$LLM_NAME" "$AGENT_NAME"; do
    id=$(instance_id "$name")
    [[ "$id" == "None" ]] || IDS+=("$id")
  done
fi
[[ ${#IDS[@]} -gt 0 ]] || exit 0
$AWS ec2 stop-instances --instance-ids "${IDS[@]}" >/dev/null
$AWS ec2 wait instance-stopped --instance-ids "${IDS[@]}"
echo "stopped ${IDS[*]}" >&2
