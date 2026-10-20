cd "$(dirname "${BASH_SOURCE[0]}")"
export AWS_PAGER=""
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would rewrite /dev/sda1

if [[ ! -f secrets.env ]]; then
  echo "missing infra/vllm-ec2/secrets.env: get it from Liam and put it here (see the top-level README)" >&2
  exit 1
fi
set -a; source config.env; source secrets.env; set +a
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
REGION=${AWS_REGION:-us-east-1}
AWS="aws --region $REGION"

if [[ -n "${AWS_ACCOUNT_ID:-}" ]]; then
  CALLER_ACCOUNT=$($AWS sts get-caller-identity --query Account --output text)
  if [[ "$CALLER_ACCOUNT" != "$AWS_ACCOUNT_ID" ]]; then
    echo "your AWS credentials are for account $CALLER_ACCOUNT, but AWS_ACCOUNT_ID in config.env is $AWS_ACCOUNT_ID" >&2
    exit 1
  fi
fi
