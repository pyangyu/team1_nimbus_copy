#!/usr/bin/env bash
set -euo pipefail
SHA=$1 ROLE=$2 UPSTREAM=${3:-}
need_tools
SRC=/opt/brownout/src
mkdir -p /opt/brownout /var/lib/brownout
rm -rf $SRC && mkdir -p $SRC
aws s3 cp --region "$REGION" --quiet "s3://$BUCKET/ci/$SHA/src.tgz" /tmp/src.tgz
tar xzf /tmp/src.tgz -C $SRC
[[ -x /opt/brownout/venv/bin/python ]] || uv venv -q --python 3.12 /opt/brownout/venv
PY=/opt/brownout/venv/bin/python
if [[ $ROLE == jev ]]; then
  uv pip install -q --python $PY -e $SRC/classifier -e $SRC/gateway
  umask 077
  echo "LIFELINE_API_KEY=$(api_key)" > /etc/brownout-gateway.env
  umask 022
  if [[ "${BROWNOUT_KEEP_STATE:-0}" != 1 ]]; then rm -f /var/lib/brownout/classifier.npz; fi
  [[ ! -f /var/lib/brownout/gateway.jsonl ]] || mv /var/lib/brownout/gateway.jsonl "/var/lib/brownout/gateway-$(date +%s).jsonl"
  cat > /etc/systemd/system/brownout-gateway.service <<UNIT
[Unit]
After=network-online.target open-jev.service
[Service]
EnvironmentFile=/etc/brownout-gateway.env
ExecStart=$PY -m brownout_gateway.cli --upstream $UPSTREAM --host 0.0.0.0 --port 9000 --jev-url http://127.0.0.1:8791/v1/systemone --warmup ${BROWNOUT_WARMUP:-20} --state /var/lib/brownout/classifier.npz --log /var/lib/brownout/gateway.jsonl
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable brownout-gateway >/dev/null 2>&1
  systemctl restart brownout-gateway
  if ! wait_http http://127.0.0.1:9000/health 30; then
    journalctl -u brownout-gateway --no-pager -n 60
    exit 1
  fi
  KEY=$(api_key)
  curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:9000/v1/models | head -c 300; echo
else
  uv pip install -q --python $PY -e "$SRC/agent[test,math]"
fi
echo "deployed $ROLE at $SHA"
