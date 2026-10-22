#!/usr/bin/env bash
set -euo pipefail
OPENJEV_COMMIT=bd4118882f733574a3250a4b65fe4d884130c08b
JEV=/opt/openjev
MARK=$JEV/.ready-$OPENJEV_COMMIT
need_tools
if systemctl list-unit-files vllm.service >/dev/null 2>&1 && systemctl is-enabled vllm >/dev/null 2>&1; then
  systemctl disable --now vllm || true
  docker rm -f vllm >/dev/null 2>&1 || true
  echo "disabled the old vLLM service"
fi
if [[ ! -f $MARK ]]; then
  mkdir -p $JEV /opt/hf
  [[ -d $JEV/Open-Jev ]] || git clone -q https://github.com/Zefan-Cai/Open-Jev.git $JEV/Open-Jev
  git -C $JEV/Open-Jev fetch -q origin
  git -C $JEV/Open-Jev checkout -q $OPENJEV_COMMIT
  [[ -x $JEV/venv/bin/python ]] || uv venv -q --python 3.12 $JEV/venv
  uv pip install -q --python $JEV/venv/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
  uv pip install -q --python $JEV/venv/bin/python "transformers==5.10.2" "peft==0.19.1" "accelerate==1.13.0" \
    "safetensors==0.8.0" "flash-linear-attention==0.5.2" "einops==0.8.2" hf_transfer huggingface-hub
  uv pip install -q --python $JEV/venv/bin/python --no-deps -e $JEV/Open-Jev
  HF_HOME=/opt/hf HF_HUB_ENABLE_HF_TRANSFER=1 $JEV/venv/bin/python - <<'PY'
import json
from huggingface_hub import snapshot_download as s
s("ZefanCai/Open-Jev-2B", allow_patterns=["package/checkpoint/*"], local_dir="/opt/openjev/models/Open-Jev-2B")
cfg = json.load(open("/opt/openjev/models/Open-Jev-2B/package/checkpoint/model.json"))
s(cfg["model_id"], revision=cfg["revision"])
print("downloaded", cfg["model_id"], cfg["revision"])
PY
  touch $MARK
  echo "installed Open-Jev $OPENJEV_COMMIT"
fi
cat > /etc/systemd/system/open-jev.service <<UNIT
[Unit]
After=network-online.target
[Service]
Environment=HF_HOME=/opt/hf
WorkingDirectory=$JEV/Open-Jev
ExecStart=$JEV/venv/bin/python -m jev.server --checkpoint $JEV/models/Open-Jev-2B/package/checkpoint --host 127.0.0.1 --port 8791
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable open-jev >/dev/null 2>&1
systemctl is-active --quiet open-jev || systemctl restart open-jev
if ! wait_http http://127.0.0.1:8791/health 60; then
  journalctl -u open-jev --no-pager -n 60
  echo "Open-Jev did not become healthy"
  exit 1
fi
curl -s http://127.0.0.1:8791/health; echo
t0=$(date +%s.%N)
curl -sf -H 'Content-Type: application/json' http://127.0.0.1:8791/v1/systemone -d '{"model":"open-jev","state":{"request":"Prove that n^3 - n is divisible by 6."},"questions":{"tier":{"type":"score","instructions":"How many thinking tokens will the model use?","criteria":["0-32","32-128","128-256","256-512","512-1K","1K-2K","2K-4K","4K-8K","8K-16K","16K+"]}}}' \
  | head -c 400; echo
echo "probe took $(echo "$(date +%s.%N) - $t0" | bc) s"
