# Brownout gateway

The front door of Thinking Brownout. The math agent (or any client using the same
`X-Lifeline-*` headers) talks to the gateway instead of vLLM. For each think request
it does five things:
1. starts the request at its requested budget with a heuristic cut-protection priority;
2. forwards it to vLLM as a stream;
3. at the same time, queues it for Jev, which answers bucket and priority in one call
   while vLLM prefills and starts decoding; the answer sets the request's budget and
   priority on arrival;
4. under load, cuts the request's thinking while it is still decoding;
5. records the online word model's bucket beside Jev's, as the comparison point.

When the request finishes, its thinking length goes back to the classifier as a label.

```
agent ──> gateway :9000 ──────────────────────────────> vLLM :8000
            | Jev queue (beside prefill/decode)            | /metrics -> pressure p
            | word model (comparison, shadow)              |
            | controller: limit = max(base x (1 - p x w), used + 64), only shrinks
            v
          Open-Jev :8791, :8792 (same GPU box, own venv; batching servers, see ../jevserve)
```

## Requests

| Request | What the gateway does |
|---|---|
| Think phase (`X-Lifeline-Phase: think`, streaming) | Predicts the bucket and priority, queues the request for Jev, then streams from vLLM with `max_tokens` = the current limit. It cuts the request if the limit drops, and calls `observe` at the end. |
| Answer phase, budget-drop continuations, anything else | Passed through unchanged. |
| The first call of a new workflow (`X-Lifeline-Workflow` not seen before) | Goes through admission first (`reject` and `defer` policies). Later calls of an admitted workflow are never rejected, so a rate limiter turns away new work, not work that is half done. |

The think response carries these headers (sent before Jev answers):
- `X-Lifeline-Tier`: `pending` (or the word model's bucket with `--predictor online`).
- `X-Lifeline-Priority`: the priority.
- `X-Lifeline-Budget`: the starting limit.

The last stream event carries `lifeline: {tier, online_tier, jev_tier, jev_at, predictor,
priority, priority_source, budget, limit, cut, cut_from, announced, notices, policy}`: `tier` is the
bucket that set the budget (`pending` if Jev never answered), `online_tier` and `jev_tier`
each predictor's bucket, and `jev_at` the thinking tokens decoded when Jev's answer landed
(0 = still queued or prefilling at vLLM). The math agent stores it per phase in `calls.jsonl`.

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
- `--predictor` sets what the `brownout` budget comes from: `jev` (default; beside the LLM),
  `jev-serial` (the gateway waits for Jev, up to `--serial-timeout`, before sending the request
  to vLLM), or `online`, the word model, as the comparison point. With `online`, Jev still answers every request, but
  its bucket and priority are only recorded. Without a Jev URL, the word model is used.
- `--notice` sets how a cut reaches the model:
  - `silent`: the request stops at the new limit with `finish_reason: length`, and the agent appends its early-stop line.
  - `announced`: the gateway immediately continues the request with `<budget_cut left=N>` appended and `max_tokens = N`. The client sees one stream. Each later drop of at least `max(--renotice-min, --renotice-frac x N)` (default `max(64, 0.1 N)`, N from the last notice) sends a new notice with the new N; a smaller drop stops the request at the limit as in `silent`. `announced` is true after the first notice, `notices` counts them (per request in the log and the stream, in total in `/v1/stats`).

## Priority (cut protection)

| Priority | Weight w | Assigned when |
|---|---|---|
| high | 0.5 | Interactive traffic, or under 20 s to the deadline |
| normal | 1.0 | Otherwise |
| low | 1.5 | Retry rounds (`solve.1`, `verify.1`, ...) |

Each request starts with this heuristic, and Jev replaces it (see below). Only `brownout`
uses the weights.

## Jev

Jev is prefill-only, so one call takes about as long as the LLM's prefill (~0.08 s warm
on its own A10G). The gateway never waits for it:
- On arrival, the request is queued for Jev and sent to vLLM in the same moment. Jev's
  answer (bucket and priority in one call) usually lands during decode, which is where
  the controller cuts, so it sets the decode budget the cut works from.
- When Jev is busy, requests wait in the Jev queue instead of being dropped. The next one
  served is the request with the fewest decoded tokens, oldest first, so requests still
  waiting in vLLM's queue (0 tokens) get Jev's answer before they start decoding.
- Requests that finish before their turn are dropped from the queue.
- Until Jev answers, the budget is the requested one, as in `uniform`. The answer lowers or
  raises it, but a running limit only ever shrinks.

`--jev-concurrency` (default 1) sets how many Jev calls run at once. `GET /v1/stats`
reports under `jev`: answers that landed before the first token vs during decode, tier
changes, priority changes, drops, and the mean wait in the queue.

## Run

```
pip install -e ../classifier -e .
LIFELINE_API_KEY=... brownout-gateway --upstream http://<vllm>:8000/v1 --host 0.0.0.0 \
  --jev-url http://127.0.0.1:8791/v1/systemone,http://127.0.0.1:8792/v1/systemone --jev-concurrency 8 \
  --state classifier.npz --log gateway.jsonl
```

Endpoints:
- `/v1/completions` and `/v1/models`, OpenAI-compatible.
- `GET /v1/stats`: gateway counters, controller, pressure histogram, priority, and classifier.
- `POST /admin/policy` with `{policy, notice, predictor, reject_waiting, admit_rate, admit_burst, reset}`; `reset: true` also clears the classifier (word model, Jev examples, stats). It refills the bucket and forgets which workflows were admitted, so a sweep can reuse workflow ids.

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
- `--predictor online`: the word model sets budgets, Jev only recorded, both scored
- `--predictor jev-serial`: the budget is Jev's before vLLM sees the request; the timeout
- reset
- Jev running beside decode and cutting a request mid-flight
- the Jev queue serving waiting requests first and dropping finished ones
- admin and auth
- upstream errors
- the real math-agent client through the gateway, including a cut and a 429

The AWS end-to-end run is in [`../infra/brownout`](../infra/brownout/README.md).
