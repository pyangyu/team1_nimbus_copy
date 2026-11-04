from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

from budget_classifier import BudgetClassifier, Buckets, JevPrior
from budget_classifier.tailfeat import FeatureJudge

from .controller import MODES, NOTICES, POLICIES, PREDICTORS, Controller
from .pressure import PressureMonitor
from .proxy import DEFAULT_NOTICE, NOTICE_TEXTS, Gateway, create_app


def build(args: argparse.Namespace) -> tuple[Gateway, str | None]:
    key = os.environ.get(args.key_env) or None
    buckets = Buckets(tuple(int(e) for e in args.edges.split(","))) if args.edges else Buckets()
    jev = JevPrior(buckets, args.jev_url, api_key=os.environ.get(args.jev_key_env) or None,
                   timeout=args.jev_timeout, max_inflight=args.jev_concurrency,
                   label=args.labels) if args.jev_url else None
    clf = BudgetClassifier(buckets, jev=jev, quantile=args.quantile, active_at=args.active_at)
    if args.state and Path(args.state).exists():
        clf.load(args.state)
    http = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), limits=httpx.Limits(max_connections=1000))
    upstreams = [u.strip().rstrip("/") for u in args.upstream.split(",") if u.strip()]
    metrics_urls = ([u.strip() for u in args.metrics_url.split(",")] if args.metrics_url
                    else [u.removesuffix("/v1") + "/metrics" for u in upstreams])
    if len(metrics_urls) != len(upstreams):
        raise SystemExit("--metrics-url needs one URL per --upstream")
    slo = args.tpot_slo_ms / 1000 if args.tpot_slo_ms else None
    monitors = [PressureMonitor(m, interval=args.poll, http=http, tpot_slo=slo) for m in metrics_urls]
    controller = Controller(monitors[0], policy=args.policy, notice=args.notice, predictor=args.predictor,
                            mode=args.cut, buckets=buckets,
                            min_left=args.min_left, max_frac=args.max_frac, active_at=args.active_at,
                            renotice_min=args.renotice_min, renotice_frac=args.renotice_frac,
                            exit_at=args.exit_at, bank=args.bank)
    gw = Gateway(upstreams, clf, controller, monitors[0], monitors=monitors, upstream_key=key, client_key=key, log_path=args.log,
                 reject_waiting=args.reject_waiting, defer_timeout=args.defer_timeout,
                 admit_rate=args.admit_rate, admit_burst=args.admit_burst,
                 notice_text=NOTICE_TEXTS.get(args.notice_text, args.notice_text), serial_timeout=args.serial_timeout, http=http,
                 labels=args.labels, probe_concurrency=args.probe_concurrency, probe_edges=args.probe_edges,
                 judge_from=args.judge_from, judge_every=args.judge_every, judge_signal=args.judge_signal,
                 judge_always=args.judge_always, judge_backend=args.judge,
                 judge_feat=FeatureJudge.load(args.judge_weights))
    return gw, args.state


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="brownout-gateway")
    ap.add_argument("--upstream", required=True, help="vLLM base URL, e.g. http://10.0.0.5:8000/v1; "
                    "several comma-separated replicas: each new request goes to the least loaded")
    ap.add_argument("--metrics-url", help="default: each upstream host's /metrics (comma list, one per upstream)")
    ap.add_argument("--key-env", default="LIFELINE_API_KEY", help="env var with the API key (clients and vLLM)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--policy", choices=POLICIES, default="brownout")
    ap.add_argument("--notice", choices=NOTICES, default="silent")
    ap.add_argument("--predictor", choices=PREDICTORS, default="jev",
                    help="what sets the tier budget: Jev beside the LLM, Jev before the LLM (jev-serial), "
                         "the online word model (comparison), or oracle (the thinking length this prompt used "
                         "in an earlier uncut run)")
    ap.add_argument("--cut", choices=MODES, default="proportional",
                    help="how brownout spreads a cut: the same fraction for every running request, ranked "
                         "(the same total, taken first from the requests whose predicted need is already met), "
                         "or settled (ranked, with Jev re-asked mid-decode whether each request's answer is settled)")
    ap.add_argument("--judge-from", type=int, default=256, help="settled: first judgment after this many thinking tokens")
    ap.add_argument("--judge-every", type=int, default=512, help="settled: judge again every this many tokens")
    ap.add_argument("--judge-signal", choices=("settled", "done", "mean"), default="settled",
                    help="settled: which of Jev's answers is P(settled)")
    ap.add_argument("--judge-always", action="store_true", help="judge in every policy (logged; only settled cuts use it)")
    ap.add_argument("--judge", choices=("jev", "feat"), default="jev",
                    help="settled: who judges the thinking so far: Jev on its own GPU, or the trained "
                         "surface-feature model (budget_classifier.tailfeat) inline on this CPU")
    ap.add_argument("--judge-weights", help="feat: weights json (default: the one shipped with budget_classifier)")
    ap.add_argument("--bank", action="store_true",
                    help="ranked/settled: keep finished requests' shed credit in a bank (sheds less under sustained pressure)")
    ap.add_argument("--exit-at", type=float, default=None,
                    help="settled: under pressure, cut a request outright once P(settled) reaches this")
    ap.add_argument("--labels", choices=("length", "needed"), default="length",
                    help="what a finished request teaches: the thinking length it used, or the smallest "
                         "budget at which its answer was already the same (found by probing the thinking)")
    ap.add_argument("--probe-concurrency", type=int, default=2, help="needed labels: probe calls in flight")
    ap.add_argument("--probe-edges", type=int, default=3, help="needed labels: bucket edges probed per request")
    ap.add_argument("--serial-timeout", type=float, default=10.0, help="jev-serial: longest wait for Jev")
    ap.add_argument("--notice-text", default=DEFAULT_NOTICE,
                    help=f"announced: text with {{N}} for the tokens left, or a preset: {', '.join(NOTICE_TEXTS)} "
                         "(countdown is the notice the training/ models learned)")
    ap.add_argument("--reject-waiting", type=float, default=8)
    ap.add_argument("--defer-timeout", type=float, default=60.0)
    ap.add_argument("--admit-rate", type=float, default=0.25, help="reject: new workflows per second")
    ap.add_argument("--admit-burst", type=float, default=8, help="reject: bucket size")
    ap.add_argument("--min-left", type=int, default=64)
    ap.add_argument("--max-frac", type=float, default=0.9)
    ap.add_argument("--renotice-min", type=int, default=64,
                    help="announced: a new notice needs the limit to drop by at least this many tokens")
    ap.add_argument("--renotice-frac", type=float, default=0.1,
                    help="announced: or this fraction of the budget left at the last notice, if larger")
    ap.add_argument("--active-at", type=float, default=0.05)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--tpot-slo-ms", type=float,
                    help="decode-time SLO per output token of a running request: pressure is also "
                         "(tpot - slo) / slo, from 0 at the SLO to 1 at twice it (default: the queue only)")
    ap.add_argument("--edges")
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--state")
    ap.add_argument("--log")
    ap.add_argument("--jev-url", help="one Jev endpoint, or several comma-separated; each call goes to the least busy")
    ap.add_argument("--jev-key-env", default="JEV_API_KEY")
    ap.add_argument("--jev-timeout", type=float, default=2.0)
    ap.add_argument("--jev-concurrency", type=int, default=1, help="Jev calls in flight at once")
    args = ap.parse_args(argv)

    import uvicorn

    gw, state = build(args)
    uvicorn.run(create_app(gw, state_path=state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    cli()
