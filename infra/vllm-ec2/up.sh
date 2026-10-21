#!/usr/bin/env bash
# Bring up (or restart) the vLLM servers on EC2, one GPU per tier (rt, batch).
# Usage: ./up.sh [rt] [batch]   (default: both) -> prints LIFELINE_*_BASE_URL / LIFELINE_API_KEY
# Settings in config.env. Credentials: secrets.env (not in git, get it from Liam), or any AWS
# profile/role for the account (CI uses an OIDC role); the vLLM API key is read from SSM if not set
set -euo pipefail
source "$(dirname "$0")/env.sh"

TIERS=("$@")
[[ ${#TIERS[@]} -gt 0 ]] || TIERS=(rt batch)
for tier in "${TIERS[@]}"; do
  [[ "$tier" == rt || "$tier" == batch ]] || { echo "unknown tier: $tier (use rt or batch)" >&2; exit 1; }
done

INSTANCE_TYPE=${INSTANCE_TYPE:-g5.xlarge}
MODEL=${MODEL:-Qwen/Qwen3-8B}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.11.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
RT_MAX_NUM_SEQS=${RT_MAX_NUM_SEQS:-8}
SG_NAME=lifeline-vllm-sg

MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
VPC=$($AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

SG=$($AWS ec2 describe-security-groups --filters Name=group-name,Values=$SG_NAME Name=vpc-id,Values=$VPC \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG" == "None" ]]; then
  SG=$($AWS ec2 create-security-group --group-name $SG_NAME --vpc-id "$VPC" \
    --description "vLLM for lifeline test agent" --query GroupId --output text)
fi
OLD=$($AWS ec2 describe-security-groups --group-ids "$SG" \
  --query 'SecurityGroups[0].IpPermissions[?ToPort==`8000`].IpRanges[].CidrIp' --output text)
[[ " $OLD " == *" $MY_IP/32 "* ]] || \
  $AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 8000 --cidr "$MY_IP/32" >/dev/null

AMI=""
ensure_instance() {
  local tier=$1 name=lifeline-vllm-$1 extra="" id state ud
  if [[ "$tier" == rt ]]; then extra="--max-num-seqs $RT_MAX_NUM_SEQS"; fi
  id=$($AWS ec2 describe-instances --filters Name=tag:Name,Values=$name \
    Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)

  if [[ "$id" == "None" ]]; then
    if [[ -z "$AMI" ]]; then
      AMI=$($AWS ec2 describe-images --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" Name=state,Values=available \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
    fi
    ud=userdata.$tier.generated.sh
    cat > "$ud" <<EOF
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
  --api-key $LIFELINE_API_KEY --enable-prefix-caching $extra
[Install]
WantedBy=multi-user.target
UNIT
mkdir -p /opt/hf
systemctl daemon-reload
systemctl enable --now vllm
EOF
    id=$($AWS ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
      --security-group-ids "$SG" --user-data file://$ud \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name},{Key=project,Value=lifeline},{Key=tier,Value=$tier}]" \
      --query 'Instances[0].InstanceId' --output text)
    echo "launched $name $id ($INSTANCE_TYPE, $MODEL)" >&2
  else
    state=$($AWS ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
    if [[ "$state" == "stopping" ]]; then $AWS ec2 wait instance-stopped --instance-ids "$id"; state=stopped; fi
    if [[ "$state" == "stopped" ]]; then $AWS ec2 start-instances --instance-ids "$id" >/dev/null; echo "started $name $id" >&2; fi
  fi
  IDS+=("$id")
}

IDS=()
for tier in "${TIERS[@]}"; do ensure_instance "$tier"; done
$AWS ec2 wait instance-running --instance-ids "${IDS[@]}"

BASE_URL=""
for i in "${!TIERS[@]}"; do
  tier=${TIERS[$i]} id=${IDS[$i]}
  IP=$($AWS ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "waiting for $tier vLLM at http://$IP:8000 (first boot downloads the model, ~5-10 min)" >&2
  up=0
  for _ in $(seq 1 120); do
    if curl -sf -m 5 -H "Authorization: Bearer $LIFELINE_API_KEY" "http://$IP:8000/v1/models" >/dev/null; then up=1; break; fi
    sleep 10
  done
  if [[ $up == 0 ]]; then echo "$tier vLLM did not come up within 20 min; instance $id is still running" >&2; exit 1; fi
  TIER_UPPER=$(echo "$tier" | tr '[:lower:]' '[:upper:]')
  echo "LIFELINE_${TIER_UPPER}_BASE_URL=http://$IP:8000/v1"
  if [[ -z "$BASE_URL" || "$tier" == batch ]]; then BASE_URL=http://$IP:8000/v1; fi
done
echo "LIFELINE_BASE_URL=$BASE_URL"
echo "LIFELINE_API_KEY=$LIFELINE_API_KEY"
