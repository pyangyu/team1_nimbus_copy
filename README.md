# team1_nimbus: Lifeline

Deadline-aware two-tier LLM serving. Requests carry deadlines; a gateway routes
interactive chat to a reserved real-time tier and agentic workflows to a
throughput-optimized batch tier, and a decision controller promotes at-risk
requests or cuts their thinking budget.

| Component | Status | Where |
|---|---|---|
| Agentic workflow (test agent, FinQA) | first version | [`agent/`](agent/) |
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

1. Copy the env file: `cp infra/vllm-ec2/.env.example infra/vllm-ec2/.env`
2. Fill in the three blank values (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `LIFELINE_API_KEY`) with what Liam sends you privately. Everything else is already set.
3. Start the servers (or find them already running) and export their URLs:
   `eval "$(infra/vllm-ec2/up.sh | sed 's/^/export /')"`
4. Stop them when you're done: `infra/vllm-ec2/down.sh` (or `down.sh rt` / `down.sh batch`).

Rules:
- `.env` is gitignored. Never commit it, and never paste the keys into an issue, PR or
  comment: everyone in the course org can read this repo.
- The servers are shared. Stopping one stops it for everyone, so check the team chat first.
- Each instance costs about $1/hour while running, so don't leave them on overnight.
  Stopped, each keeps a 120 GB disk for about $10/month.
- `up.sh` adds your current IP to the firewall; if your IP changes, run it again.
