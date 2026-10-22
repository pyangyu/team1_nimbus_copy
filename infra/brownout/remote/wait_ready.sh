#!/usr/bin/env bash
set -euo pipefail
LLM_URL=$1
need_tools
KEY=$(api_key)
echo "waiting for vLLM at $LLM_URL"
wait_http "$LLM_URL/models" 120 "$KEY" || { echo "vLLM did not answer within 20 min"; exit 1; }
echo "vLLM ready"
