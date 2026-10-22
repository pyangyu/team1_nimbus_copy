#!/usr/bin/env bash
# One-time: S3 bucket, internal security group, the agent box, and the rt GPU repurposed as the Jev box.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if ! $AWS s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  $AWS s3api create-bucket --bucket "$BUCKET" >/dev/null
  $AWS s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  $AWS s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
    '{"Rules":[{"ID":"expire-ci","Status":"Enabled","Filter":{"Prefix":"ci/"},"Expiration":{"Days":30}}]}' >/dev/null
  echo "created bucket $BUCKET" >&2
fi

VPC=$($AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$($AWS ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG" == "None" ]]; then
  SG=$($AWS ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC" \
    --description "Thinking Brownout boxes talk to each other (vLLM 8000, gateway 9000)" --query GroupId --output text)
  for port in 8000 9000; do
    $AWS ec2 authorize-security-group-ingress --group-id "$SG" \
      --ip-permissions "IpProtocol=tcp,FromPort=$port,ToPort=$port,UserIdGroupPairs=[{GroupId=$SG}]" >/dev/null
  done
  echo "created $SG_NAME $SG" >&2
fi

join_box() {
  local id=$1 groups assoc
  assoc=$($AWS ec2 describe-iam-instance-profile-associations --filters "Name=instance-id,Values=$id" \
    --query 'IamInstanceProfileAssociations[0].AssociationId' --output text)
  if [[ "$assoc" == "None" ]]; then
    $AWS ec2 associate-iam-instance-profile --instance-id "$id" --iam-instance-profile "Name=$PROFILE_NAME" >/dev/null
  fi
  groups=$($AWS ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].SecurityGroups[].GroupId' --output text)
  if [[ " $groups " != *" $SG "* ]]; then
    # shellcheck disable=SC2086
    $AWS ec2 modify-instance-attribute --instance-id "$id" --groups $groups "$SG"
  fi
}

JEV=$(instance_id "$JEV_NAME")
if [[ "$JEV" == "None" ]]; then
  JEV=$(instance_id lifeline-vllm-rt)
  [[ "$JEV" != "None" ]] || { echo "no lifeline-vllm-rt instance to repurpose" >&2; exit 1; }
  $AWS ec2 create-tags --resources "$JEV" --tags "Key=Name,Value=$JEV_NAME" Key=tier,Value=jev
  echo "repurposed lifeline-vllm-rt $JEV as $JEV_NAME" >&2
fi
join_box "$JEV"

LLM=$(instance_id "$LLM_NAME")
[[ "$LLM" != "None" ]] || { echo "no $LLM_NAME instance (run infra/vllm-ec2/up.sh batch once)" >&2; exit 1; }
join_box "$LLM"

AGENT=$(instance_id "$AGENT_NAME")
if [[ "$AGENT" == "None" ]]; then
  AMI=$($AWS ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query Parameter.Value --output text)
  AGENT=$($AWS ec2 run-instances --image-id "$AMI" --instance-type t3.small \
    --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE_NAME" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$AGENT_NAME},{Key=project,Value=lifeline},{Key=tier,Value=agent}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "launched $AGENT_NAME $AGENT (t3.small)" >&2
fi
echo "JEV=$JEV LLM=$LLM AGENT=$AGENT SG=$SG"
