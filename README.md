# team1_nimbus: Thinking Brownout

Rate-limit the reasoning, not the users. When a burst overloads an LLM service, a
rate limiter rejects requests, a queue lets latency grow, and scale-out arrives
minutes after the burst. Thinking Brownout serves everyone a little cheaper
instead: as the queue backs up, it shrinks the thinking budget of every request
already decoding in proportion to the pressure, tells the model about the cut,
and trains the model to finish well with less.

```
requests -> budget classifier -> vLLM queue -> running batch (trained reasoning model)
                (tier per request)      |               ^
                                        v               | shrink budgets + <budget_cut left=N>
                                 pressure monitor -> budget controller
                                 (backlog -> p)       (budget x (1 - p))
finished requests -> actual thinking length -> online update of the classifier
```

1. **Budget classifier.** A lightweight Jev-style classifier predicts a thinking
   bucket (ten literal buckets, 0-32 tokens up to 16K+) for each prompt. It runs silently,
   logging until load rises, and learns online: each finished request's actual
   thinking length is a free label, so it needs no offline training and follows
   model upgrades.
2. **Budget controller.** `p = backlog / batch capacity`. Every in-flight request's
   budget becomes `max(tier budget x (1 - p), used + 64)`, and the engine injects a
   `<budget_cut left=N>` token at the next step so the model wraps up on its own
   terms instead of being force-closed.
3. **Trained model.** SFT teaches the model what the cut token means; RL (GRPO)
   with cuts injected at random times and sizes teaches the trade-off: commit to
   one approach, skip redundant re-checks, finish cleanly with no reasoning leaking
   past `</think>`, and use the budget that is left.

What is new is the combination: a load-driven cut on requests already decoding,
announced to the model, with the model trained for it. Load-driven budgets exist
only at admission, and budget-trained models only see budgets fixed per request.

The workload is math (MATH-500, AIME 2026, GSM8K): answers are checked
automatically, so RL is cheap, and bursts look like a tutoring service on an
exam night.

| Component | Status | Where |
|---|---|---|
| Math agent: multi-step workflows with deadlines; client-side budget drop and notice for experiments | first version | [`agent/`](agent/) (FinQA version in [`agent/archive/finqa/`](agent/archive/finqa/)) |
| vLLM on EC2 (Qwen3-8B, one A10G) | scripts ready, CI runs on it | [`infra/vllm-ec2/`](infra/vllm-ec2/) |
| Brownout on AWS: agent box -> Jev GPU (Open-Jev-2B + gateway) -> vLLM GPU, policy sweep in CI | first version | [`infra/brownout/`](infra/brownout/) |
| Budget-drop baselines and SFT/RL training (Qwen3-4B + LoRA) | first version | [`training/`](training/) |
| Budget classifier (online, ten buckets, Jev predicts on every request, online word model kept as the comparison) | first version | [`classifier/`](classifier/) |
| Open-Jev serving under bursts: several Open-Jev servers on the Jev GPU, each micro-batching concurrent calls | first version | [`jevserve/`](jevserve/) |
| Gateway: pressure monitor, budget controller (in-flight cuts, silent or announced), cut-protection priority, Jev beside prefill with a queue for busy periods, baselines (none, reject, defer, uniform) | first version, in front of vLLM | [`gateway/`](gateway/) |
| Cut-token injection inside the engine | todo (the gateway emulates it by aborting and continuing the stream) | |

The request headers the agent sends (budgets, deadlines, budget drops) are in
[`agent/README.md`](agent/README.md#request-headers).

## GPUs on the shared AWS account

The team shares two g5.xlarge instances (one A10G each) in Liam's AWS account:
`lifeline-vllm-batch` runs vLLM (Qwen3-8B), and `brownout-jev` (formerly `lifeline-vllm-rt`) runs
Open-Jev-2B and the Brownout gateway. A t3.small, `brownout-agent`, runs the math agent in the
end-to-end tests. See [`infra/brownout/`](infra/brownout/) for that setup.

1. Get `secrets.env` from Liam and put it in `infra/vllm-ec2/` (or, if you have your own
   credentials for the account, skip it: the scripts use your AWS profile and read the vLLM
   API key from SSM `/capstone/vllm-api-key`). Settings are in
   `infra/vllm-ec2/config.env`; `AWS_ACCOUNT_ID` there is Liam's account, so to use your
   own AWS account, change it and use your own `secrets.env`. The scripts refuse to run if
   your credentials belong to a different account.
2. Start the servers (or find them already running) and export their URLs:
   `eval "$(infra/vllm-ec2/up.sh | sed 's/^/export /')"`
3. Stop them when you're done: `infra/vllm-ec2/down.sh` (vLLM only) or `infra/brownout/down.sh` (all three boxes).

CI has two suites. **Smoke** (unit tests, secret scan, shellcheck; under 5 minutes) runs on
every push to `main`. **Full** adds the AWS GPU jobs below and runs only on demand:
`gh workflow run CI -f suite=full` (or Actions → CI → Run workflow → full).

In the full suite, job `aws` does the same as the steps above through an
OIDC role (`capstone-github-ci`, no stored keys): it starts the batch GPU, runs the agent tests
against it, and stops it again unless someone already had it running.
Job `brownout-e2e` then runs the whole Brownout chain (agent box, gateway, Open-Jev and vLLM),
sweeps the policies, uploads `brownout-results` as an artifact, and stops the boxes it started.

Rules:
- `secrets.env` is gitignored. Never commit it or paste its contents into an issue, PR or
  comment: everyone in the course org can read this repo.
- The servers are shared. Stopping one stops it for everyone, so check the team chat first.
- Each instance costs about $1/hour while running, so don't leave them on overnight.
  Stopped, each keeps a 120 GB disk for about $10/month.
- `up.sh` adds your current IP to the firewall; if your IP changes, run it again.
