# team1_nimbus: Lifeline

Deadline-aware two-tier LLM serving. Requests carry deadlines; a gateway routes
interactive chat to a reserved real-time tier and agentic workflows to a
throughput-optimized batch tier, and a decision controller promotes at-risk
requests or cuts their thinking budget.

| Component | Status | Where |
|---|---|---|
| Agentic workflow (math agent: MATH-500, AIME 2026, GSM8K) | first version | [`agent/`](agent/) (FinQA version in [`agent/archive/finqa/`](agent/archive/finqa/)) |
| vLLM on EC2: two GPUs, one per tier (rt, batch) | scripts ready | [`infra/vllm-ec2/`](infra/vllm-ec2/) |
| LifeLine Gateway | todo | |
| RT tier / Batch tier (vLLM pods on Kubernetes) | todo | |
| Metrics collector (queue, KV cache, TTFT, TPOT) | todo | |
| Decision controller | todo | |

The header contract between the agent and the gateway is in
[`agent/README.md`](agent/README.md#contract-with-the-gateway).

## GPUs on the shared AWS account

The team shares two g5.xlarge instances (one A10G each) in Liam's AWS account:
`lifeline-vllm-rt` for the real-time tier and `lifeline-vllm-batch` for the batch tier.

1. Get `secrets.env` from Liam and put it in `infra/vllm-ec2/` (or, if you have your own
   credentials for the account, skip it: the scripts use your AWS profile and read the vLLM
   API key from SSM `/capstone/vllm-api-key`). Settings are in
   `infra/vllm-ec2/config.env`; `AWS_ACCOUNT_ID` there is Liam's account, so to use your
   own AWS account, change it and use your own `secrets.env`. The scripts refuse to run if
   your credentials belong to a different account.
2. Start the servers (or find them already running) and export their URLs:
   `eval "$(infra/vllm-ec2/up.sh | sed 's/^/export /')"`
3. Stop them when you're done: `infra/vllm-ec2/down.sh` (or `down.sh rt` / `down.sh batch`).

CI (`.github/workflows/ci.yml`, job `aws`) does the same on every push to `main` through an
OIDC role (`capstone-github-ci`, no stored keys): it starts the batch GPU, runs the agent tests
against it, and stops it again unless someone already had it running.

Rules:
- `secrets.env` is gitignored. Never commit it or paste its contents into an issue, PR or
  comment: everyone in the course org can read this repo.
- The servers are shared. Stopping one stops it for everyone, so check the team chat first.
- Each instance costs about $1/hour while running, so don't leave them on overnight.
  Stopped, each keeps a 120 GB disk for about $10/month.
- `up.sh` adds your current IP to the firewall; if your IP changes, run it again.
