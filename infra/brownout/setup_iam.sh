#!/usr/bin/env bash
# One-time, with an admin profile: instance role for the brownout boxes and SSM/S3 rights for the CI role.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if ! aws iam get-role --role-name "$PROFILE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$PROFILE_NAME" --assume-role-policy-document file://iam/ec2-trust.json >/dev/null
fi
aws iam attach-role-policy --role-name "$PROFILE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name "$PROFILE_NAME" --policy-name brownout-ec2 --policy-document file://iam/brownout-ec2.json
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$PROFILE_NAME"
fi
aws iam put-role-policy --role-name capstone-github-ci --policy-name brownout-e2e --policy-document file://iam/ci-brownout.json
aws iam update-role --role-name capstone-github-ci --max-session-duration 14400
echo "IAM ready: instance profile $PROFILE_NAME, CI policy brownout-e2e" >&2
