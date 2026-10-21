# Test agent: first runs (2026-10-16)

## What was run

| | |
|---|---|
| Hardware | 1x g5.xlarge (NVIDIA A10G 24 GB), us-east-1 |
| Server | vLLM 0.11.0, `--max-model-len 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching` |
| Model | Qwen/Qwen3-8B, bf16 |
| Data | FinQA dev split; questions sampled with seed 0 |
| Agent config | k=8 (+8 per round), thinking budget 1024 doubling up to 4096, up to 3 rounds, confidence threshold 0.7 |

Three runs, all against the same single vLLM server (no tiers yet):

1. **Solo:** 50 questions, one at a time. This gives each question's uncontended
   latency, which is used to set deadlines in the load runs.
2. **Load 0.05/s:** 100 workflows, Poisson arrivals at 0.05 workflows/s, drawn from the
   50 solo questions.
3. **Load 0.2/s:** the same at 0.2 workflows/s.

Each load workflow's deadline is `arrival + slack x (its solo latency)`, where slack is
drawn uniformly from {1.5, 3, 6}.

Commands:

```bash
python -m lifeline_agent.run --n 50 --out runs/solo
python -m lifeline_agent.loadgen --rate 0.05 --n 100 --solo-ref runs/solo/workflows.jsonl --only-ref
python -m lifeline_agent.loadgen --rate 0.2  --n 100 --solo-ref runs/solo/workflows.jsonl --only-ref
```

## Results

| Run | n | Accuracy (strict / lenient) | Latency p50 / p90 / p99 | Deadline met | Correct and on time |
|---|---|---|---|---|---|
| Solo | 50 | 0.38 / 0.52 | 40 / 166 / 264 s | n/a | n/a |
| Load 0.05/s | 100 | 0.42 / 0.52 | 44 / 133 / 195 s | 0.96 | 0.40 |
| Load 0.2/s | 100 | 0.43 / 0.54 | 59 / 171 / 336 s | 0.83 | 0.37 |

Deadline met, by slack:

| Slack | Load 0.05/s | Load 0.2/s |
|---|---|---|
| 1.5x | 0.92 (n=36) | 0.56 (n=36) |
| 3x | 0.96 (n=25) | 0.96 (n=25) |
| 6x | 1.00 (n=39) | 1.00 (n=39) |

Per-call behavior (from `calls.jsonl`):

| Run | LLM calls | TTFT p50 / p99 | Thinking decode, tokens/s (p50) | Reasoning calls that hit the budget |
|---|---|---|---|---|
| Solo | 176 | 0.33 / 0.49 s | 28.2 | 39 / 73 |
| Load 0.05/s | 340 | 0.37 / 0.54 s | 25.5 | 73 / 134 |
| Load 0.2/s | 344 | 0.42 / 11.7 s | 18.8 | 75 / 135 |

Solo workflows by number of rounds:

| Rounds | n | Latency p50 | Accuracy | Thinking tokens (mean) |
|---|---|---|---|---|
| 1 | 35 | 39 s | 0.40 | 826 |
| 2 | 7 | 91 s | 0.57 | 2277 |
| 3 | 8 | 194 s | 0.12 | 5201 |

## Observations

1. **Contention costs deadlines before it costs accuracy.**
   - Going from 0.05 to 0.2 workflows/s leaves accuracy flat (0.42 to 0.43).
   - Tight-deadline workflows (1.5x slack) drop from 92% to 56% on time.
   - Per-token decode slows from 25.5 to 18.8 tokens/s, and the TTFT tail grows from
     0.5 s to 11.7 s as requests queue.
   - This is the regime Lifeline targets: promoting the at-risk workflows to a faster
     tier should recover the deadlines without changing answers.
2. **Late workflows are often wrong anyway.**
   - At 0.2/s, 11 of 17 late workflows were also wrong.
   - So a controller that values on-time *correct* answers can skip rescuing some of them.
   - That needs a predictor of correctness, which is where Jev fits.
3. **Cost per workflow is very uneven.**
   - p90 latency is about 4x p50, driven by rounds.
   - Three-round workflows take about 5x the time of one-round ones and are rarely
     correct (1 of 8).
   - Cutting the budget or stopping early on those is the "cut budget" lever, and it
     looks cheap in accuracy.
4. **The budget binds often.**
   - About half of reasoning calls hit the thinking budget (39 of 73 solo).
   - Budget size directly trades latency for accuracy, so it is a real control knob.
5. **The CHECK step is poorly calibrated.**
   - 43 of 53 solo checks answered 0.85 or 0.95.
   - 16 of 23 retries were triggered by DSL errors, not low confidence.
   - A better confidence signal (Jev) would decide retries better.
6. **Accuracy is limited partly by retrieval.**
   - BM25 finds all gold evidence for 69% of dev questions at k=8.
   - Common answer errors: a percent given instead of a fraction (the lenient score
     accepts these), sign errors, and running out of budget.
   - These matter less for the scheduling experiments, which compare policies on the
     same questions.

## Cost

The instance ran 2 h 05 min on demand (about $2.10). It is stopped now. Its 120 GB
gp3 volume (about $0.32/day) keeps the model cache, so a restart with
`infra/vllm-ec2/up.sh` takes a few minutes instead of about 18.

## Next

- Gateway in front of two vLLM tiers: RT (reserved) and batch. Route by
  `X-Lifeline-Class`.
- Re-run these exact loads (same seed, same solo reference) through the gateway as the
  baseline, then with promotion and budget cuts.
- Add interactive chat traffic on the RT tier so promotions have an interference cost.

## Files

Each run folder has `workflows.jsonl` (one row per question), `calls.jsonl` (one row
per LLM call) and `summary.json`; the load runs also have `config.json`. The solo
`workflows.jsonl` includes a full per-step trace for each question.
