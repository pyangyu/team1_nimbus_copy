# jevserve: serving Open-Jev under bursts

The gateway asks Open-Jev once per think request. In a burst those calls arrive faster than one
model can answer them. Open-Jev's own server (`jev.server`) scores every request under one lock,
so calls are answered strictly one at a time. In the 2026-10-22 sweep this left GSM8K calls
waiting 2–3 s in the gateway's Jev queue (MATH-500, with half the arrival rate: about 0.3 s).

Two fixes, used together:

1. **Several Open-Jev servers on one GPU.** The Jev box runs `open-jev@<port>.service` once per
   port in `JEV_PORTS` (default `8791 8792`, set in `infra/brownout/remote/common.sh`). Open-Jev-2B
   is about 4–5 GB in bf16, so two copies fit on the A10G's 24 GB with room for activations.
   The gateway gets every URL (`--jev-url a,b`) and sends each call to the server with the
   fewest calls in flight (`JevPrior` in `classifier/budget_classifier/jev.py`).
2. **Micro-batching inside each server** (`batch_server.py`). Requests are queued; the server
   waits up to `--window-ms` (default 5 ms) for more, up to `--max-requests` (default 16), and
   scores all their candidate sequences together in forward passes of up to `--batch-size`
   sequences. Answers are identical to Open-Jev's own server; only the scheduling changes.

The gateway keeps `--jev-concurrency` calls in flight across all servers (`JEV_CONCURRENCY`,
default 8), so each server has requests to batch.

## Run

`batch_server.py` needs Open-Jev's venv, which provides the `jev` package:

```
/opt/openjev/venv/bin/python jevserve/batch_server.py \
  --checkpoint /opt/openjev/models/Open-Jev-2B/package/checkpoint --port 8791
```

It serves the same endpoints as Open-Jev (`/v1/systemone`, `/v1/inference`, `/api/jev`), plus
`/health`, which also reports how many batches and requests it has scored.

On AWS this is automatic: `setup_jev.sh` installs Open-Jev and the `open-jev@.service` template,
and `deploy.sh` (role `jev`) starts one server per port from the deployed commit, probes each,
then starts the gateway with all of their URLs.

## Measuring it

- Gateway `/v1/stats`: `jev.mean_wait_s` (time in the gateway's queue) and
  `classifier.jev.mean_call_s` (time per Jev HTTP call, including the server's batching window).
- The live sweep (`agent/tests/test_brownout_live.py`) writes both per policy as
  `jev_mean_wait_s` and `jev_mean_call_s` in `results.md`.
- Each server's `/health` is saved by `collect_jev.sh` as `jev/open-jev-health.txt`.

## Knobs

| Setting | Where | Default |
|---|---|---|
| Servers | `JEV_PORTS` in `infra/brownout/remote/common.sh` | `8791 8792` |
| Calls in flight from the gateway | `JEV_CONCURRENCY`, same file | 8 |
| Batching window | `--window-ms` on `batch_server.py` | 5 ms |
| Requests per batch | `--max-requests` | 16 |
| Sequences per forward pass | `--batch-size` | 32 |

If a server runs out of GPU memory, lower `--batch-size` or drop to one port.
