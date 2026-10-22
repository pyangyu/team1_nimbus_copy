#!/usr/bin/env bash
set -euo pipefail
SHA=$1 TAG=$2
need_tools
KEY=$(api_key)
curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:9000/v1/stats > /tmp/gateway-stats.json || true
journalctl -u brownout-gateway --no-pager -n 300 > /tmp/gateway-journal.log || true
journalctl -u open-jev --no-pager -n 300 > /tmp/open-jev-journal.log || true
for f in /tmp/gateway-stats.json /tmp/gateway-journal.log /tmp/open-jev-journal.log /var/lib/brownout/gateway.jsonl; do
  [[ ! -f $f ]] || aws s3 cp --region "$REGION" --quiet "$f" "s3://$BUCKET/ci/$SHA/results/$TAG/jev/$(basename "$f")"
done
echo "collected gateway logs"
