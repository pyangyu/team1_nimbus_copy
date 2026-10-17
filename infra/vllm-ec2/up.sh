#!/usr/bin/env bash
# Bring up (or restart) one vLLM server on EC2 for the test agent.
# Usage: ./up.sh            -> prints LIFELINE_BASE_URL / LIFELINE_API_KEY
# Config via env: INSTANCE_TYPE, MODEL, VLLM_IMAGE, MAX_MODEL_LEN, AWS_REGION
set -euo pipefail
cd "$(dirname "$0")"
export AWS_PAGER=""
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would rewrite /dev/sda1

NAME=lifeline-vllm
INSTANCE_TYPE=${INSTANCE_TYPE:-g5.xlarge}
MODEL=${MODEL:-Qwen/Qwen3-8B}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.11.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
REGION=${AWS_REGION:-us-east-1}
AWS="aws --region $REGION"

# API key lives only in the local, gitignored .env
if [[ ! -f .env ]]; then
  echo "LIFELINE_API_KEY=$(python -c 'import secrets;print(secrets.token_hex(16))')" > .env
fi
source .env

MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
VPC=$($AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

SG=$($AWS ec2 describe-security-groups --filters Name=group-name,Values=$NAME-sg Name=vpc-id,Values=$VPC \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG" == "None" ]]; then
  SG=$($AWS ec2 create-security-group --group-name $NAME-sg --vpc-id "$VPC" \
    --description "vLLM for lifeline test agent" --query GroupId --output text)
fi
# Only the current caller's IP may reach port 8000
OLD=$($AWS ec2 describe-security-groups --group-ids "$SG" \
  --query 'SecurityGroups[0].IpPermissions[?ToPort==`8000`].IpRanges[].CidrIp' --output text)
for cidr in $OLD; do
  [[ "$cidr" == "$MY_IP/32" ]] || $AWS ec2 revoke-security-group-ingress --group-id "$SG" --protocol tcp --port 8000 --cidr "$cidr" >/dev/null
done
[[ " $OLD " == *" $MY_IP/32 "* ]] || \
  $AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 8000 --cidr "$MY_IP/32" >/dev/null

ID=$($AWS ec2 describe-instances --filters Name=tag:Name,Values=$NAME \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

if [[ "$ID" == "None" ]]; then
  AMI=$($AWS ec2 describe-images --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" Name=state,Values=available \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  UD=userdata.generated.sh
  cat > $UD <<EOF
#!/bin/bash
cat > /etc/systemd/system/vllm.service <<UNIT
[Unit]
After=docker.service
Requires=docker.service
[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f vllm
ExecStart=/usr/bin/docker run --name vllm --gpus all --ipc=host -p 8000:8000 \
  -v /opt/hf:/root/.cache/huggingface $VLLM_IMAGE \
  --model $MODEL --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.92 \
  --api-key $LIFELINE_API_KEY --enable-prefix-caching
[Install]
WantedBy=multi-user.target
UNIT
mkdir -p /opt/hf
systemctl daemon-reload
systemctl enable --now vllm
EOF
  ID=$($AWS ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --security-group-ids "$SG" --user-data file://$UD \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=lifeline}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "launched $ID ($INSTANCE_TYPE, $MODEL)" >&2
else
  STATE=$($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$STATE" == "stopping" ]]; then $AWS ec2 wait instance-stopped --instance-ids "$ID"; STATE=stopped; fi
  if [[ "$STATE" == "stopped" ]]; then $AWS ec2 start-instances --instance-ids "$ID" >/dev/null; echo "started $ID" >&2; fi
fi

$AWS ec2 wait instance-running --instance-ids "$ID"
IP=$($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "waiting for vLLM at http://$IP:8000 (first boot downloads the model, ~5-10 min)" >&2
for _ in $(seq 1 120); do
  if curl -sf -m 5 -H "Authorization: Bearer $LIFELINE_API_KEY" "http://$IP:8000/v1/models" >/dev/null; then
    echo "LIFELINE_BASE_URL=http://$IP:8000/v1"
    echo "LIFELINE_API_KEY=$LIFELINE_API_KEY"
    exit 0
  fi
  sleep 10
done
echo "vLLM did not come up within 20 min; instance $ID is still running" >&2
exit 1
