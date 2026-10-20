# Lifeline test agent

The "Agentic workflow" box in the architecture diagram. It answers FinQA questions
(numeric QA over S&P 500 annual reports) as multi-step LLM workflows, so every
workflow has a verifiable answer, a deadline, and a token cost. Its traffic is what
the Lifeline Gateway and decision controller will schedule.

## Workflow

```
PLAN -> RETRIEVE -> REASON -> EXECUTE -> CHECK --(confident or out of rounds)--> ANSWER
           ^                                |
           +---- widen k, double budget ----+
```

| Step | Kind | What it does |
|---|---|---|
| PLAN | LLM, no thinking | Lists the facts needed (line items, years) |
| RETRIEVE | local | BM25 over every FinQA page of that filing (company/year), top k |
| REASON | LLM, thinking budget | Writes a FinQA DSL program, e.g. `subtract(5829, 5735), divide(#0, 5735)` |
| EXECUTE | local | Deterministic DSL calculator (`dsl.py`, matches gold `exe_ans` on 100% of train/dev/test) |
| CHECK | LLM, no thinking | Confidence in [0, 1]; below `--conf` triggers another round |

Scoring: `accuracy` is FinQA execution accuracy; `accuracy_lenient` also accepts 1%
error and percent-vs-fraction answers.

BM25 recall of all gold evidence on dev: 55% at k=4, 69% at k=8, 77% at k=16, 82% at k=24.
So the retry loop does real work, and per-workflow cost varies a lot.

## Contract with the gateway

The agent only knows one OpenAI-compatible `base_url` (`/v1/completions`, streaming).
Point it at vLLM today and at the Lifeline Gateway later. Every request carries:

| Header | Meaning |
|---|---|
| `X-Lifeline-Class` | `agentic` (interactive chat traffic will send `interactive`) |
| `X-Lifeline-Workflow` | workflow id, shared by all calls of one question |
| `X-Lifeline-Step` | `plan`, `reason.<round>`, `check.<round>` |
| `X-Lifeline-Phase` | `think` or `answer` (a reasoning call is two requests) |
| `X-Lifeline-Thinking-Budget` | thinking tokens allowed for this call |
| `X-Lifeline-Deadline` | workflow deadline, epoch ms |
| `X-Lifeline-Tier` | optional tier hint; the gateway may echo the tier it chose in the response header |

The thinking budget is enforced client-side by budget forcing (Qwen3 chat format):
the think phase stops at `</think>` or at the budget, then the answer phase is a second
request on the same prefix (cheap with prefix caching). If the budget ran out, a short
"give the answer now" line is inserted before `</think>`. This is the "cut budget" lever
from the diagram: the controller can lower the budget of a later call without touching
the agent.

## Running

```bash
cd agent
pip install -e .[test] && pytest -q

# two GPU boxes (rt + batch tier) running vLLM + Qwen3-8B; needs infra/vllm-ec2/.env (see the top-level README)
eval "$(../infra/vllm-ec2/up.sh | sed 's/^/export /')"   # LIFELINE_BASE_URL points at the batch tier

python -m lifeline_agent.run --index 0 -v            # one question, full trace
python -m lifeline_agent.run --n 50 --out runs/solo  # solo calibration
python -m lifeline_agent.loadgen --rate 0.1 --n 100 --solo-ref runs/solo/workflows.jsonl --only-ref

../infra/vllm-ec2/down.sh                            # stop both GPU boxes
```

FinQA is downloaded to `agent/data/` on first use. Each run writes `calls.jsonl`
(one row per LLM call: tokens, TTFT, latency, budget, phases), `workflows.jsonl`
(one row per question: answer, correctness, deadline met, totals) and `summary.json`.

Load generator: open-loop Poisson arrivals at `--rate` workflows/s. Each workflow's
deadline is `arrival + slack * (that question's solo latency)`, with slack drawn from
`--slack` (default `1.5,3,6`).

## First results (2026-10-16)

Full writeup and raw outputs: [`results/2026-10-16/`](results/2026-10-16/WRITEUP.md).

One g5.xlarge (A10G 24 GB), vLLM 0.11.0, Qwen3-8B bf16, default config
(k=8, thinking budget 1024 doubling to 4096, up to 3 rounds, confidence 0.7).
Single-stream decode is about 27 tokens/s, so a 1024-token thought takes about 38 s.

| Run | n | Accuracy (strict / lenient) | Latency p50 / p90 / p99 | Deadline met | Met, slack 1.5x |
|---|---|---|---|---|---|
| Solo (one at a time) | 50 | 0.38 / 0.52 | 40 / 166 / 264 s | n/a | n/a |
| Load, 0.05 workflows/s | 100 | 0.42 / 0.52 | 44 / 133 / 195 s | 0.96 | 0.92 |
| Load, 0.2 workflows/s | 100 | 0.43 / 0.54 | 59 / 171 / 336 s | 0.83 | 0.56 |

Notes:
- Per-workflow cost is very uneven (p90 is about 4x p50): hard questions exhaust the
  budget, fail the check or the DSL, and go another round with a larger budget.
- On one GPU, going from 0.05 to 0.2 workflows/s drops tight-deadline (1.5x) workflows
  from 92% to 56% on time. That gap is what tiering and promotion should close.
- The CHECK step is poorly calibrated: 43 of 53 solo checks answered 0.85 or 0.95, and
  16 of 23 retries came from DSL errors rather than low confidence. That makes it a
  candidate for Jev.
- Common misses: the answer given as a percent instead of a fraction (counted by lenient
  scoring), sign errors, and running out of thinking budget.

## Layout

```
lifeline_agent/
  datasets/      Example/Chunk types, scorer; finqa.py loader + per-filing corpus
  dsl.py         FinQA program executor
  retrieval.py   BM25
  llm.py         completions client, headers, budget forcing, per-call timing
  prompts.py
  workflow.py    the state machine above
  run.py         sequential runs (trace / calibration)
  loadgen.py     open-loop load with deadlines
  metrics.py
tests/           DSL + scorer; workflow against a fake streaming vLLM
```

TAT-QA can be added as another module in `datasets/` that produces the same
`Example`/`Chunk` types.
