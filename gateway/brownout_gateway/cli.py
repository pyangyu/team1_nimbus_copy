from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

from budget_classifier import BudgetClassifier, Buckets, JevPrior

from .controller import MODES, NOTICES, POLICIES, PREDICTORS, Controller
from .pressure import PressureMonitor
from .proxy import DEFAULT_NOTICE, Gateway, create_app


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
    metrics_url = args.metrics_url or args.upstream.rstrip("/").removesuffix("/v1") + "/metrics"
    monitor = PressureMonitor(metrics_url, interval=args.poll, http=http)
    controller = Controller(monitor, policy=args.policy, notice=args.notice, predictor=args.predictor,
                            mode=args.cut, buckets=buckets,
                            min_left=args.min_left, max_frac=args.max_frac, active_at=args.active_at)
    gw = Gateway(args.upstream, clf, controller, monitor, upstream_key=key, client_key=key, log_path=args.log,
                 reject_waiting=args.reject_waiting, defer_timeout=args.defer_timeout,
                 admit_rate=args.admit_rate, admit_burst=args.admit_burst,
                 notice_text=args.notice_text, serial_timeout=args.serial_timeout, http=http,
                 labels=args.labels, probe_concurrency=args.probe_concurrency, probe_edges=args.probe_edges)
    return gw, args.state


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="brownout-gateway")
    ap.add_argument("--upstream", required=True, help="vLLM base URL, e.g. http://10.0.0.5:8000/v1")
    ap.add_argument("--metrics-url", help="default: the upstream host's /metrics")
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
                    help="how brownout spreads a cut: the same fraction for every running request, or ranked "
                         "(the same total, taken first from the requests whose predicted need is already met)")
    ap.add_argument("--labels", choices=("length", "needed"), default="length",
                    help="what a finished request teaches: the thinking length it used, or the smallest "
                         "budget at which its answer was already the same (found by probing the thinking)")
    ap.add_argument("--probe-concurrency", type=int, default=2, help="needed labels: probe calls in flight")
    ap.add_argument("--probe-edges", type=int, default=3, help="needed labels: bucket edges probed per request")
    ap.add_argument("--serial-timeout", type=float, default=10.0, help="jev-serial: longest wait for Jev")
    ap.add_argument("--notice-text", default=DEFAULT_NOTICE)
    ap.add_argument("--reject-waiting", type=float, default=8)
    ap.add_argument("--defer-timeout", type=float, default=60.0)
    ap.add_argument("--admit-rate", type=float, default=0.25, help="reject: new workflows per second")
    ap.add_argument("--admit-burst", type=float, default=8, help="reject: bucket size")
    ap.add_argument("--min-left", type=int, default=64)
    ap.add_argument("--max-frac", type=float, default=0.9)
    ap.add_argument("--active-at", type=float, default=0.05)
    ap.add_argument("--poll", type=float, default=0.5)
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
