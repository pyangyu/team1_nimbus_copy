cd "$(dirname "${BASH_SOURCE[0]}")"
export AWS_PAGER=""
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would rewrite /dev/sda1

if [[ ! -f .env ]]; then
  echo "missing infra/vllm-ec2/.env: cp .env.example .env and fill in the blanks (see the top-level README)" >&2
  exit 1
fi
set -a; source .env; set +a
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
REGION=${AWS_REGION:-us-east-1}
AWS="aws --region $REGION"
