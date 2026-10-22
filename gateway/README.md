# Brownout gateway

The front door of Thinking Brownout. The math agent (or any client using the same
`X-Lifeline-*` headers) talks to the gateway instead of vLLM. For each think request
it does four things:
1. asks the budget classifier for a thinking bucket;
2. assigns a cut-protection priority;
3. forwards the request to vLLM as a stream;
4. under load, cuts the request's thinking while it is still decoding.

When the request finishes, its thinking length goes back to the classifier as a label.

```
agent ──> gateway :9000 ──────────────────────────────> vLLM :8000
            | classifier (online; Open-Jev while cold)     | /metrics -> pressure p
            | priority (heuristic now, Jev in background)  |
            | controller: limit = max(base x (1 - p x w), used + 64), only shrinks
            v
          Open-Jev :8791 (same GPU box, own venv)
```

## Requests

| Request | What the gateway does |
|---|---|
| Think phase (`X-Lifeline-Phase: think`, streaming) | Predicts the bucket and priority, then streams from vLLM with `max_tokens` = the current limit. It cuts the request if the limit drops, and calls `observe` at the end. |
| Answer phase, budget-drop continuations, anything else | Passed through unchanged. |
| The first call of a new workflow (`X-Lifeline-Workflow` not seen before) | Goes through admission first (`reject` and `defer` policies). Later calls of an admitted workflow are never rejected, so a rate limiter turns away new work, not work that is half done. |

The think response carries these headers:
- `X-Lifeline-Tier`: the bucket label.
- `X-Lifeline-Priority`: the priority.
- `X-Lifeline-Budget`: the starting limit.

The last stream event carries `lifeline: {tier, priority, budget, limit, cut, cut_from,
announced, policy}`. The math agent stores it per phase in `calls.jsonl`.

## Policies (`--policy`, or `POST /admin/policy` at runtime)

| Policy | Under load |
|---|---|
| `none` | Nothing. The queue grows. |
| `reject` | Classic rate limiter: a token bucket on new workflows (`--admit-rate`, default 0.25/s, about what one A10G finishes with no limiting; `--admit-burst` 8). 429 when the bucket is empty. |
| `defer` | Holds new calls at the gateway until the queue drains; 429 after `defer_timeout`. |
| `uniform` | Brownout without the classifier or priority: `limit = max(requested x (1 - p), used + 64)`. |
| `brownout` | Brownout with both: `base = min(requested, classifier budget)` and `w` by priority. |

- Pressure is `p = min(1, waiting / running)` from vLLM `/metrics`, polled every 0.5 s.
- Nothing is cut below `p = 0.05` (shadow mode).
- `--notice` sets how a cut reaches the model:
  - `silent`: the request stops at the new limit with `finish_reason: length`, and the agent appends its early-stop line.
  - `announced`: the gateway immediately continues the request with `<budget_cut left=N>` appended and `max_tokens = N`. The client sees one stream.

## Priority (cut protection)

| Priority | Weight w | Assigned when |
|---|---|---|
| high | 0.5 | Interactive traffic, or under 20 s to the deadline |
| normal | 1.0 | Otherwise |
| low | 1.5 | Retry rounds (`solve.1`, `verify.1`, ...) |

Each request starts with this heuristic. While the classifier still asks Jev about the
bucket, the same Jev call carries a three-level priority question. After that, a
background Jev call refines the priority without blocking the request. At most one of
these runs at a time, and the rest are dropped. Only `brownout` uses the weights.

## Run

```
pip install -e ../classifier -e .
LIFELINE_API_KEY=... brownout-gateway --upstream http://<vllm>:8000/v1 --host 0.0.0.0 \
  --jev-url http://127.0.0.1:8791/v1/systemone --state classifier.npz --log gateway.jsonl
```

Endpoints:
- `/v1/completions` and `/v1/models`, OpenAI-compatible.
- `GET /v1/stats`: gateway counters, controller, pressure histogram, priority, and classifier.
- `POST /admin/policy` with `{policy, notice, reject_waiting, admit_rate, admit_burst}`. It refills the bucket.

`gateway.jsonl` has one row per think request, prompt included, so
`budget-classifier replay` can run on real traffic.

## Tests

`pytest -q` runs against a fake vLLM that streams tokens slowly and exposes `/metrics`:
- passthrough
- tier and observe
- a censored label when the budget is hit
- `reject`, `defer` and its timeout
- an in-flight cut, both silent and announced (upstream aborted, continuation carries the notice)
- no cut without pressure
- low priority cut harder than high, and `uniform` ignoring priority
- tier and priority in one Jev call
- the background refiner
- admin and auth
- upstream errors
- the real math-agent client through the gateway, including a cut and a 429

The AWS end-to-end run is in [`../infra/brownout`](../infra/brownout/README.md).
