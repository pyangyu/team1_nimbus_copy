# shellcheck shell=bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export AWS_PAGER=""
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would rewrite /dev/sda1

set -a
source config.env
if [[ -f secrets.env ]]; then source secrets.env; fi
set +a
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
REGION=${AWS_REGION:-us-east-1}
AWS="aws --region $REGION"

if ! CALLER_ACCOUNT=$($AWS sts get-caller-identity --query Account --output text 2>/dev/null); then
  echo "no AWS credentials: put secrets.env (from Liam) in infra/vllm-ec2/, or use a profile/role for account ${AWS_ACCOUNT_ID:-?}" >&2
  exit 1
fi
if [[ -n "${AWS_ACCOUNT_ID:-}" && "$CALLER_ACCOUNT" != "$AWS_ACCOUNT_ID" ]]; then
  echo "your AWS credentials are for account $CALLER_ACCOUNT, but AWS_ACCOUNT_ID in config.env is $AWS_ACCOUNT_ID" >&2
  exit 1
fi
if [[ -z "${LIFELINE_API_KEY:-}" ]]; then
  LIFELINE_API_KEY=$($AWS ssm get-parameter --name "$API_KEY_PARAM" --with-decryption \
    --query Parameter.Value --output text) || { echo "cannot read $API_KEY_PARAM from SSM" >&2; exit 1; }
  export LIFELINE_API_KEY
fi
