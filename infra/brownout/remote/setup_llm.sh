#!/usr/bin/env bash
set -euo pipefail
IMAGE=$1 MODEL=$2 MAX_MODEL_LEN=$3
need_tools
for unit in $(systemctl list-units --all --plain --no-legend 'open-jev@*' | awk '{print $1}'); do
  systemctl disable --now "$unit" >/dev/null 2>&1 || true
done
systemctl stop vllm >/dev/null 2>&1 || true
docker rm -f vllm >/dev/null 2>&1 || true
docker container prune -f >/dev/null || true
rm -rf /opt/hf/xet /root/.cache/pip
df -h / | tail -n 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
umask 077
echo "LIFELINE_API_KEY=$(api_key)" > /etc/brownout-vllm.env
umask 022
mkdir -p /opt/hf
cat > /etc/systemd/system/vllm.service <<UNIT
[Unit]
After=docker.service
Requires=docker.service
[Service]
EnvironmentFile=/etc/brownout-vllm.env
Restart=always
ExecStartPre=-/usr/bin/docker rm -f vllm
ExecStart=/usr/bin/docker run --name vllm --gpus all --ipc=host -p 8000:8000 \
  -e HF_HUB_DISABLE_XET=1 -v /opt/hf:/root/.cache/huggingface $IMAGE \
  --model $MODEL --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.92 \
  --api-key \${LIFELINE_API_KEY} --enable-prefix-caching
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable vllm >/dev/null 2>&1
systemctl restart vllm
KEY=$(api_key)
if ! wait_http http://127.0.0.1:8000/v1/models 180 "$KEY"; then
  journalctl -u vllm --no-pager -n 80
  echo "vLLM on this box did not answer within 30 min"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
df -h / | tail -n 1
echo "vLLM $MODEL ready on 127.0.0.1:8000"
