#!/usr/bin/env bash
export HOME=/root PATH=/root/.local/bin:/snap/bin:/usr/local/bin:$PATH
REGION=us-east-1
BUCKET=capstone-ci-826671498662
export REGION BUCKET
JEV_PORTS=${JEV_PORTS:-"8791 8792"}
JEV_CONCURRENCY=${JEV_CONCURRENCY:-8}
JEV_TIMEOUT=${JEV_TIMEOUT:-10}
need_tools() {
  command -v aws >/dev/null || snap install aws-cli --classic >/dev/null
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
}
api_key() {
  aws ssm get-parameter --region "$REGION" --name /capstone/vllm-api-key --with-decryption --query Parameter.Value --output text
}
wait_http() {
  local url=$1 tries=$2 auth=${3:-}
  for _ in $(seq 1 "$tries"); do
    if [[ -n "$auth" ]]; then curl -sf -m 5 -H "Authorization: Bearer $auth" "$url" >/dev/null && return 0
    else curl -sf -m 5 "$url" >/dev/null && return 0; fi
    sleep 10
  done
  return 1
}
