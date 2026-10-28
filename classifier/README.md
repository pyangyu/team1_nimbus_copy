# Budget classifier

Piece 1 of Thinking Brownout. It predicts how long each request will think, as one of
ten literal buckets. The budget controller scales that budget when the queue backs up.
Jev makes the prediction; an online word model makes its own beside it as the comparison
point. Every finished request's actual thinking length is a free label: it trains the
word model, calibrates Jev (as in-context examples), and scores both on the same requests.

```
request ──> /v1/predict ──> bucket + budget (logged; enforced only under pressure)
                ^
                | Jev (prediction) + features of the prompt -> online softmax (comparison)
finished ──> /v1/observe (thinking tokens used, finished or cut) ──> one online update
```

## Buckets

| Bucket | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Thinking tokens | 0-32 | 32-128 | 128-256 | 256-512 | 512-1K | 1K-2K | 2K-4K | 4K-8K | 8K-16K | 16K+ |
| Budget given | 32 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |

The budget is the bucket's upper edge. Edges are configurable (`--edges`); 2 to 10
buckets are allowed, because a Jev `score` question takes at most 10 levels.

## How it predicts

1. **Online word model (comparison).** Hashed word and bigram features of the prompt, plus length and
   digit-count features, go into a softmax over the buckets. It is trained by AdaGrad,
   one step per labeled request. Targets are soft over neighboring buckets, since being
   one bucket off is a small error. It never sets budgets in the gateway unless the
   gateway runs with `--predictor online`.
2. **Jev on every request (the prediction).** If a Jev endpoint is configured (hosted Jev or a local Open-Jev,
   same `/v1/systemone` wire format), it asks one `score` question with the ten buckets
   as levels. The state includes the prompt and the most similar recent finished requests
   with the thinking tokens they used, so Jev also calibrates to the live model.
   `predict_online` answers at once from the word model; `ask_jev` asks Jev for the same
   request later (the gateway runs it beside the LLM's prefill). `predict` does both and
   returns Jev's answer, or the word model's if Jev failed. The two are never mixed.
   For each finished request that both predicted, `paired` in the stats counts exact,
   within-one and under-predicted buckets and the probability given to the true bucket,
   for each source.
3. **Pick.** The bucket is the `quantile` point of the distribution (default 0.5, the
   median). Raise it to under-cut less often.

## What the label means

`JevPrior(label=...)` picks the wording Jev sees:
- `length`: the thinking tokens the model used (the objective, the `thinking_tokens_used` field in
  the examples, and the question all say so).
- `needed`: the smallest budget at which the answer stopped changing. The gateway produces these
  labels by probing (`brownout-gateway --labels needed`); the classifier itself is unchanged, it
  just learns whatever it is told.

## How it learns

| Request ended | Label | Update |
|---|---|---|
| Model closed `</think>` itself | exact bucket | full update, counts toward accuracy |
| Cut by the controller or hit its budget | "at least this bucket" (censored) | pushes probability only to that bucket and above (`--censored update`, default) or is skipped (`--censored skip`) |

Censored updates matter once budgets are enforced: requests the classifier
under-predicted get cut, and skipping them would hide exactly those mistakes.

**Model upgrades.** Each observation carries the served model name. When it changes,
the AdaGrad state is reset so the classifier moves quickly again, the Jev mix starts
over for the new version, and accuracy is tracked per version. That is the experiment
behind the "re-converges after an upgrade" claim: replay traffic from model A, then
model B, and plot accuracy over the requests after the switch (`replay`, below).

**Shadow mode.** Predictions carry `shadow: true` while pressure is below `active_at`
(default 0.05). The controller posts pressure to `/v1/pressure`; nothing else changes.

## HTTP API (`budget-classifier serve`, port 8792)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/predict` | `request_id`, `prompt`, optional `model` | `bucket`, `label`, `budget`, `confidence`, `source` (`cold`, `online`, `jev`), `shadow`, `probs` |
| `POST /v1/observe` | `request_id`, `thinking_tokens`, `finished` (default true), optional `model`, `prompt` | the label used |
| `POST /v1/pressure` | `p` | `pressure`, `shadow` |
| `GET /v1/stats` | | per model version: labels, censored, rolling exact and within-one-bucket accuracy, under-prediction rate, mean probability each source gave the true bucket, Jev's weight |
| `POST /v1/save` | | writes `--state` (also every `--save-every` observations and on shutdown) |

```
pip install -e "classifier[server]"
budget-classifier serve --state runs/classifier.npz
budget-classifier serve --jev-url http://127.0.0.1:8791/v1/systemone   # with a local Open-Jev
```

For hosted Jev, put its URL in `--jev-url` and the key in `JEV_API_KEY`. `--jev-url` also takes
several comma-separated URLs; each call goes to the server with the fewest calls in flight
(see `../jevserve` for running several Open-Jev servers on one GPU).

## Replay (offline evaluation)

```
budget-classifier replay rows.jsonl --report-every 100 --out curve.jsonl
```

Each row is `{"prompt", "thinking_tokens", "model"?, "finished"?, "id"?}`. Every
request is predicted before its label is seen, exactly as online. Every
`--report-every` rows it writes the current version's rolling accuracy, then a final
summary. Concatenate rows from two model versions to measure re-convergence.

## Tests

`pytest -q` runs without a GPU or network:
- buckets and features
- the online model: learning, ordinal targets, censored updates
- the classifier on a synthetic workload: learning from cold, a model upgrade that
  quadruples thinking lengths (accuracy drops, then recovers within about 20
  requests), censored handling, shadow mode, quantile, save and load
- the Jev request shape (a ten-level `score` question with recent examples), the
  Jev-to-online hand-off, and fallback when Jev fails
- the HTTP API and the replay CLI
