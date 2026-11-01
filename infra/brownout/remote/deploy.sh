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
  JEV_ARGS=""
  if [[ "${BROWNOUT_JEV:-1}" == 0 ]]; then
    for unit in $(systemctl list-units --all --plain --no-legend 'open-jev@*' | awk '{print $1}'); do
      systemctl disable --now "$unit" >/dev/null 2>&1 || true
    done
    echo "Jev off: no Open-Jev servers, gateway without --jev-url"
  else
    for unit in $(systemctl list-units --all --plain --no-legend 'open-jev@*' | awk '{print $1}'); do
      port=${unit#open-jev@}
      port=${port%.service}
      [[ " $JEV_PORTS " == *" $port "* ]] || systemctl disable --now "$unit" >/dev/null 2>&1 || true
    done
    JEV_URLS=""
    for port in $JEV_PORTS; do
      systemctl enable "open-jev@$port" >/dev/null 2>&1
      systemctl restart "open-jev@$port"
      JEV_URLS+="${JEV_URLS:+,}http://127.0.0.1:$port/v1/systemone"
    done
    for port in $JEV_PORTS; do
      if ! wait_http "http://127.0.0.1:$port/health" 60; then
        journalctl -u "open-jev@$port" --no-pager -n 60
        echo "Open-Jev on :$port did not become healthy"
        exit 1
      fi
    done
    t0=$(date +%s.%N)
    for port in $JEV_PORTS; do
      curl -sf -H 'Content-Type: application/json' "http://127.0.0.1:$port/v1/systemone" -d '{"model":"open-jev","state":{"request":"Prove that n^3 - n is divisible by 6."},"questions":{"tier":{"type":"score","instructions":"How many thinking tokens will the model use?","criteria":["0-32","32-128","128-256","256-512","512-1K","1K-2K","2K-4K","4K-8K","8K-16K","16K+"]}}}' \
        | head -c 200 &
    done
    wait
    echo; echo "probes on all Jev servers took $(echo "$(date +%s.%N) - $t0" | bc) s"
    JEV_ARGS="--jev-url $JEV_URLS --jev-concurrency $JEV_CONCURRENCY --jev-timeout $JEV_TIMEOUT"
  fi
  KEY=$(api_key)
  IFS=, read -ra UPSTREAMS <<< "$UPSTREAM"
  for u in "${UPSTREAMS[@]}"; do
    wait_http "$u/models" 120 "$KEY" || { echo "vLLM at $u did not answer within 20 min"; exit 1; }
    echo "vLLM at $u ready"
  done
  umask 077
  echo "LIFELINE_API_KEY=$(api_key)" > /etc/brownout-gateway.env
  umask 022
  if [[ "${BROWNOUT_KEEP_STATE:-0}" != 1 ]]; then rm -f /var/lib/brownout/classifier.npz; fi
  [[ ! -f /var/lib/brownout/gateway.jsonl ]] || mv /var/lib/brownout/gateway.jsonl "/var/lib/brownout/gateway-$(date +%s).jsonl"
  cat > /etc/systemd/system/brownout-gateway.service <<UNIT
[Unit]
After=network-online.target
[Service]
EnvironmentFile=/etc/brownout-gateway.env
ExecStart=$PY -m brownout_gateway.cli --upstream $UPSTREAM --host 0.0.0.0 --port 9000 $JEV_ARGS ${BROWNOUT_NOTICE:+--notice-text $BROWNOUT_NOTICE} --state /var/lib/brownout/classifier.npz --log /var/lib/brownout/gateway.jsonl
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
  curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:9000/v1/models | head -c 300; echo
else
  uv pip install -q --python $PY -e "$SRC/agent[test,math]"
fi
echo "deployed $ROLE at $SHA"
