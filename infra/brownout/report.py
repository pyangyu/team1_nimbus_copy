"""Chart a spike or realistic suite run: python3 infra/brownout/report.py brownout-results [--open]

Writes report.html into the results folder: correct, on time and correct-and-on-time per policy, the vLLM queue
over the run, and latency and thinking tokens by arrival time.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
NAMES = {"none": "No control", "uniform": "Uniform cut", "brownout": "Brownout", "reject": "Reject", "defer": "Defer"}
PREDICTORS = {"online": "classifier", "jev": "Jev", "jev-serial": "Jev, waited for", "oracle": "true lengths"}
QUEUE_BIN_S = 5.0
ARRIVAL_BIN_S = 20.0


def label(spec: str) -> str:
    spec, _, model = spec.partition("@")
    policy, notice, predictor = (spec.split(":") + ["", ""])[:3]
    out = NAMES.get(policy, policy)
    if policy in ("uniform", "brownout"):
        out += ", announced" if notice == "announced" else ", silent"
    if policy == "brownout" and predictor:
        out += f", {PREDICTORS.get(predictor, predictor)}"
    return out + (f" + {model}" if model else "")


def run_dir(root: Path, spec: str) -> Path:
    spec, _, model = spec.partition("@")
    return root / f"realistic-{spec.replace(':', '-')}{'-' + model if model else ''}"


def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def vllm(d: Path, t0: float, col: int) -> list[list[float]]:
    by: dict[tuple[int, int], list[float]] = {}
    for row in jsonl(d / "pressure.jsonl"):
        by.setdefault((int((row[0] - t0) // QUEUE_BIN_S), row[-1] if len(row) > 4 else 0), []).append(row[col])
    bins: dict[int, float] = {}
    for (k, _), xs in by.items():
        bins[k] = bins.get(k, 0.0) + statistics.fmean(xs)
    return [[round(k * QUEUE_BIN_S, 1), round(v, 2)] for k, v in sorted(bins.items()) if k >= 0]


def decode(d: Path, t0: float) -> list[list[float]]:
    by: dict[int, dict[int, list[float]]] = {}
    for row in jsonl(d / "pressure.jsonl"):
        if len(row) > 5 and row[4] is not None:
            by.setdefault(int((row[0] - t0) // QUEUE_BIN_S), {}).setdefault(row[-1], []).append(row[4])
    return [[round(k * QUEUE_BIN_S, 1), round(statistics.fmean(statistics.fmean(xs) for xs in reps.values()), 1)]
            for k, reps in sorted(by.items()) if k >= 0]


def by_arrival(wfs: list[dict], t0: float, key: str) -> list[list[float]]:
    bins: dict[int, list[float]] = {}
    for w in wfs:
        if w.get(key) is not None and not w.get("cancelled"):
            bins.setdefault(int((w["arrival"] - t0) // ARRIVAL_BIN_S), []).append(w[key])
    return [[round((k + 0.5) * ARRIVAL_BIN_S, 1), round(statistics.median(xs), 1)] for k, xs in sorted(bins.items())]


def bursts(profile: str) -> list[list[float]]:
    out: list[list[float]] = []
    t = 0.0
    for seg in filter(None, profile.split(",")):
        d, r = (float(x) for x in seg.split(":"))
        if r > 1.0:
            if out and out[-1][1] == t:
                out[-1][1], out[-1][2] = t + d, max(out[-1][2], r)
            else:
                out.append([t, t + d, r])
        t += d
    return out


def paired(wfs: dict[str, dict[str, dict]]) -> list[dict]:
    ok = lambda w: bool(w.get("correct")) and bool(w.get("met_deadline"))
    specs, out = list(wfs), []
    for i, a in enumerate(specs):
        for b in specs[i + 1:]:
            x, y = wfs[a], wfs[b]
            if not x or set(x) != set(y) or any(x[k]["example_id"] != y[k]["example_id"] for k in x):
                continue
            gained = sum(ok(y[k]) and not ok(x[k]) for k in x)
            lost = sum(ok(x[k]) and not ok(y[k]) for k in x)
            n = gained + lost
            p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(gained, lost) + 1)) / 2 ** n) if n else 1.0
            out.append({"a": label(a), "b": label(b), "gained": gained, "lost": lost, "p": round(p, 3)})
    return out


def segments(profile: str) -> str:
    segs = [tuple(float(x) for x in seg.split(":")) for seg in filter(None, profile.split(","))]
    return ", then ".join(f"{r:.2f}x C for {d:.0f} s" for d, r in segs)


def setup(root: Path, res: dict) -> dict:
    d = run_dir(root, res["runs"][0]["policy"])
    cfg = json.loads((d / "config.json").read_text())
    s = json.loads((d / "summary.json").read_text())
    tr, cap, warm = s.get("trace") or {}, res.get("capacity") or {}, res.get("warmup") or {}
    slo = res.get("tpot_slo_ms")
    out = {"model": cfg.get("model"), "seed": res.get("seed"),
           "problems": f"{res['runs'][0]['n']} per policy from MATH-500 ({res.get('workload')}"
                       + (f", half {res['split']} by problem id" if res.get("split") else "") + ")",
           "capacity": f"C = {cap.get('capacity_wf_s')} workflows/s"
                       + (" (given)" if cap.get("source") == "BROWNOUT_CAPACITY" else " (measured)"),
           "load": segments(s.get("profile") or ""),
           "pressure": "the vLLM queue (waiting / running)" + (f", or decoding slower than {slo} ms per token" if slo else "")}
    if tr:
        out["arrivals"] = (f"BurstGPT {tr['log']} log, hours {tr['start_h']} to {tr['start_h'] + tr['window_h']:.2f} "
                           f"({tr['window_n']} requests), replayed {tr['compression']:.0f}x faster in "
                           f"{s.get('profile_s', 0):.0f} s and thinned at random to {s.get('arrivals')} arrivals")
    if cfg.get("level_budgets"):
        out["deadlines"] = (f"arrival + slack x solo latency; slack {cfg.get('slack', '').replace(',', ', ')} per problem; "
                            "solo latency 150 s for levels 1-2, 300 s for level 3, 450 s for levels 4-5")
        out["budgets"] = "2,048 thinking tokens for levels 1-2 (4,096 on a retry), 4,096 for level 3 (6,144), 6,144 for levels 4-5"
    if cfg.get("tail_s"):
        out["tail"] = f"workflows still running {cfg['tail_s']:.0f} s after the last arrival are cancelled and count as wrong and late"
    if warm:
        out["warmup"] = (f"{warm.get('n')} workflows from the other half of MATH-500 under no control "
                         f"({warm.get('labels')} finished think calls, {warm.get('censored')} cut or capped); the gateway saves "
                         "that state and every policy starts from it")
    return out


def collect(root: Path) -> dict:
    res = json.loads((root / "results.json").read_text())
    runs, profile, wfs_by = [], "", {}
    for k, r in enumerate(res["runs"]):
        d = run_dir(root, r["policy"])
        s = json.loads((d / "summary.json").read_text())
        wfs = jsonl(d / "workflows.jsonl")
        wfs_by[r["policy"]] = {w["workflow_id"]: w for w in wfs}
        profile = profile or r.get("profile") or ""
        runs.append({"spec": r["policy"], "name": label(r["policy"]), "light": LIGHT[k % 8], "dark": DARK[k % 8],
                     "n": r["n"], "accuracy": r["accuracy"], "deadline_met": r["deadline_met"],
                     "goodput": r["correct_and_on_time"],
                     **{key: r.get(key) for key in ("goodput_baseline", "goodput_burst", "cancelled", "latency_p50_s",
                                                    "latency_p95_s", "thinking_tokens_mean", "think", "cut", "announced",
                                                    "notices", "p_mean", "queue_frac", "tpot_ms_mean")},
                     "by_band": {b: {"n": v.get("n"), "goodput": v.get("correct_and_on_time")}
                                 for b, v in (r.get("by_band") or {}).items()},
                     "queue": vllm(d, s["t0"], 2), "running": vllm(d, s["t0"], 3), "tpot": decode(d, s["t0"]),
                     "latency": by_arrival(wfs, s["t0"], "latency_s"),
                     "thinking": by_arrival(wfs, s["t0"], "thinking_tokens")})
    cap = res.get("capacity") or {}
    return {"suite": res.get("suite"), "workload": res.get("workload"), "trace_args": res.get("trace_args"),
            "capacity": cap.get("capacity_wf_s"), "warmup": res.get("warmup") or {}, "bursts": bursts(profile),
            "profile": profile, "tpot_slo_ms": float(res["tpot_slo_ms"]) if res.get("tpot_slo_ms") else None,
            "setup": setup(root, res), "paired": paired(wfs_by), "runs": runs}


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --band: rgba(137, 135, 129, 0.10); --border: rgba(11, 11, 11, 0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --band: rgba(195, 194, 183, 0.08); --border: rgba(255, 255, 255, 0.10);
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --band: rgba(195, 194, 183, 0.08); --border: rgba(255, 255, 255, 0.10);
}
html, body { margin: 0; }
.viz-root { background: var(--page); color: var(--text-primary); min-height: 100vh;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 32px 24px 48px; box-sizing: border-box; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
.headline { font-size: 16px; margin: 0 0 24px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 20px 20px 12px;
  margin: 0 0 20px; position: relative; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.card p.note { color: var(--text-secondary); margin: 0 0 12px; font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; color: var(--text-secondary); font-size: 13px; }
.legend span.key { display: inline-block; width: 14px; height: 2px; border-radius: 1px; vertical-align: middle;
  margin-right: 6px; }
.multiples { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px 20px; }
.multiples h3 { font-size: 13px; font-weight: 600; margin: 4px 0 0; color: var(--text-secondary); }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg text.cat { fill: var(--text-secondary); font-size: 12px; font-variant-numeric: normal; }
svg text.val { fill: var(--text-primary); font-size: 12px; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  display: none; z-index: 2; min-width: 140px; }
.tip .head { color: var(--text-secondary); margin-bottom: 4px; }
.tip .row { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.tip .row b { font-weight: 600; color: var(--text-primary); min-width: 44px; font-variant-numeric: tabular-nums; }
.tip .row i { font-style: normal; color: var(--text-secondary); }
.tip .row span.key { display: inline-block; width: 12px; height: 2px; border-radius: 1px; }
details { margin: 8px 0 4px; color: var(--text-secondary); font-size: 13px; }
summary { cursor: pointer; }
table { border-collapse: collapse; margin-top: 8px; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 3px 10px 3px 0; border-bottom: 1px solid var(--grid); color: var(--text-primary); }
th { color: var(--text-secondary); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
.foot { color: var(--muted); font-size: 12px; margin-top: 16px; }
nav.tabs { display: flex; gap: 4px; margin: 10px 0 20px; border-bottom: 1px solid var(--grid); }
nav.tabs a { color: var(--text-secondary); text-decoration: none; padding: 6px 12px; margin-bottom: -1px;
  border-bottom: 2px solid transparent; }
nav.tabs a[aria-current="page"] { color: var(--text-primary); border-bottom-color: var(--text-primary); font-weight: 600; }
.scroll { overflow-x: auto; }
th.l, td.l { text-align: left; }
.card p.body { margin: 4px 0 10px; }
dl.facts { display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; margin: 8px 0; font-size: 13px; }
dl.facts dt { color: var(--text-secondary); }
dl.facts dd { margin: 0; }
</style>
</head>
<body>
<div class="viz-root"><div class="wrap" id="app"></div></div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const dark = () => getComputedStyle(document.querySelector(".viz-root")).colorScheme === "dark";
const color = r => dark() ? r.dark : r.light;
const pct = v => v == null ? "–" : Math.round(v * 100) + "%";
const fmt = v => v == null ? "–" : v >= 1000 ? Math.round(v).toLocaleString() : (Math.round(v * 10) / 10).toString();

function el(tag, attrs = {}, parent) {
  const e = tag.startsWith("svg:") ? document.createElementNS(NS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v; else if (k === "class") e.setAttribute("class", v); else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function tipBox(card) {
  const t = el("div", {class: "tip"}, card);
  return {
    show(x, y, head, rows) {
      t.replaceChildren();
      el("div", {class: "head", text: head}, t);
      for (const r of rows) {
        const row = el("div", {class: "row"}, t);
        const k = el("span", {class: "key"}, row); k.style.background = r.color;
        el("b", {text: r.value}, row);
        el("i", {text: r.name}, row);
      }
      t.style.display = "block";
      const cw = card.clientWidth, tw = t.offsetWidth;
      t.style.left = Math.min(Math.max(8, x + 14), cw - tw - 8) + "px";
      t.style.top = (y + 14) + "px";
    },
    hide() { t.style.display = "none"; }
  };
}

function card(title, note) {
  const c = el("section", {class: "card"}, document.getElementById("app"));
  el("h2", {text: title}, c);
  if (note) el("p", {class: "note", text: note}, c);
  return c;
}

function table(c, head, rows) {
  const d = el("details", {}, c);
  el("summary", {text: "Table view"}, d);
  const t = el("table", {}, d);
  const tr = el("tr", {}, el("thead", {}, t));
  for (const h of head) el("th", {text: h}, tr);
  const tb = el("tbody", {}, t);
  for (const r of rows) { const row = el("tr", {}, tb); for (const v of r) el("td", {text: v}, row); }
}

function bars(c, parent, title, key, tip) {
  const box = el("div", {}, parent);
  el("h3", {text: title}, box);
  const runs = DATA.runs, rowH = 30, right = 44, top = 6, H = top + runs.length * rowH + 22;
  const left = Math.max(110, Math.round(14 + 6.8 * Math.max(...runs.map(r => r.name.length)))), W = left + 180 + right;
  parent.style.gridTemplateColumns = `repeat(auto-fit, minmax(${W}px, 1fr))`;
  const svg = el("svg:svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title, style: `max-width: ${W}px`}, box);
  const x = v => left + (W - left - right) * v;
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    el("svg:line", {x1: x(g), x2: x(g), y1: top, y2: H - 20, stroke: "var(--grid)", "stroke-width": 1}, svg);
    el("svg:text", {x: x(g), y: H - 6, "text-anchor": "middle", text: Math.round(g * 100) + "%"}, svg);
  }
  el("svg:line", {x1: x(0), x2: x(0), y1: top, y2: H - 20, stroke: "var(--axis)", "stroke-width": 1}, svg);
  runs.forEach((r, i) => {
    const v = r[key] || 0, y = top + i * rowH + (rowH - 18) / 2, w = Math.max(0, x(v) - x(0)), h = 18, rad = Math.min(4, w);
    el("svg:text", {class: "cat", x: left - 8, y: y + 13, "text-anchor": "end", text: r.name}, svg);
    const d = `M${x(0)},${y}h${w - rad}a${rad},${rad} 0 0 1 ${rad},${rad}v${h - 2 * rad}a${rad},${rad} 0 0 1 -${rad},${rad}h-${w - rad}z`;
    const bar = el("svg:path", {d, fill: color(r), tabindex: 0}, svg);
    el("svg:text", {class: "val", x: x(v) + 6, y: y + 13, text: pct(r[key])}, svg);
    const hit = el("svg:rect", {x: 0, y: top + i * rowH, width: W, height: rowH, fill: "transparent"}, svg);
    const on = e => {
      bar.setAttribute("opacity", 0.8);
      const b = c.getBoundingClientRect();
      tip.show(e.clientX - b.left, e.clientY - b.top, title, [{color: color(r), value: pct(r[key]), name: r.name}]);
    };
    const off = () => { bar.removeAttribute("opacity"); tip.hide(); };
    hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", off);
    bar.addEventListener("focus", () => { const b = bar.getBoundingClientRect(); on({clientX: b.right, clientY: b.top}); });
    bar.addEventListener("blur", off);
  });
}

function lines(title, note, key, unit, ref) {
  const c = card(title, note), tip = tipBox(c), runs = DATA.runs;
  const lg = el("div", {class: "legend"}, c);
  for (const r of runs) { const s = el("span", {}, lg); const k = el("span", {class: "key"}, s);
    k.style.background = color(r); s.appendChild(document.createTextNode(r.name)); }
  const W = 900, H = 280, L = 52, R = 16, T = 10, B = 34;
  const xs = [...new Set(runs.flatMap(r => r[key].map(p => p[0])))].sort((a, b) => a - b);
  const xmax = Math.max(60, ...xs);
  const ymax = niceMax(Math.max(0, ref ? ref.value * 1.2 : 0, ...runs.flatMap(r => r[key].map(p => p[1]))));
  const x = v => L + (W - L - R) * v / xmax, y = v => T + (H - T - B) * (1 - v / ymax);
  const svg = el("svg:svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title}, c);
  for (const [a, b, rate] of DATA.bursts) {
    el("svg:rect", {x: x(a), y: T, width: Math.max(0, x(Math.min(b, xmax)) - x(a)), height: H - T - B,
      fill: "var(--band)"}, svg);
    el("svg:text", {x: x(a) + 6, y: T + 14, text: DATA.bursts.length > 1 ? `burst, up to ${rate.toFixed(1)}× capacity`
      : `spike, ${rate.toFixed(1)}× capacity`}, svg);
  }
  for (let k = 0; k <= 4; k++) {
    const v = ymax * k / 4;
    el("svg:line", {x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: k ? "var(--grid)" : "var(--axis)", "stroke-width": 1}, svg);
    el("svg:text", {x: L - 8, y: y(v) + 4, "text-anchor": "end", text: fmt(v)}, svg);
  }
  if (ref) {
    el("svg:line", {x1: L, x2: W - R, y1: y(ref.value), y2: y(ref.value), stroke: "var(--text-secondary)",
      "stroke-width": 1}, svg);
    el("svg:text", {x: W - R, y: y(ref.value) - 6, "text-anchor": "end", text: ref.label}, svg);
  }
  const step = xmax > 900 ? 300 : xmax > 400 ? 120 : 60;
  for (let v = 0; v <= xmax; v += step) el("svg:text", {x: x(v), y: H - 14, "text-anchor": "middle",
    text: v >= 60 ? `${Math.round(v / 60)} min` : "0"}, svg);
  el("svg:text", {x: L, y: H - 1, text: "time since the policy's run started"}, svg);
  for (const r of runs) {
    if (!r[key].length) continue;
    const d = r[key].map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join("");
    el("svg:path", {d, fill: "none", stroke: color(r), "stroke-width": 2, "stroke-linejoin": "round",
      "stroke-linecap": "round"}, svg);
  }
  const cross = el("svg:line", {y1: T, y2: H - B, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden"}, svg);
  const dots = runs.map(r => el("svg:circle", {r: 4, fill: color(r), stroke: "var(--surface-1)", "stroke-width": 2,
    visibility: "hidden"}, svg));
  const hit = el("svg:rect", {x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent"}, svg);
  hit.addEventListener("pointermove", e => {
    const b = svg.getBoundingClientRect(), sx = (e.clientX - b.left) * W / b.width;
    const xv = (sx - L) / (W - L - R) * xmax;
    const near = xs.reduce((a, v) => Math.abs(v - xv) < Math.abs(a - xv) ? v : a, xs[0]);
    cross.setAttribute("x1", x(near)); cross.setAttribute("x2", x(near)); cross.setAttribute("visibility", "visible");
    const rows = [];
    runs.forEach((r, i) => {
      const p = r[key].find(q => q[0] === near);
      if (p) { dots[i].setAttribute("cx", x(p[0])); dots[i].setAttribute("cy", y(p[1]));
        dots[i].setAttribute("visibility", "visible"); rows.push({color: color(r), value: fmt(p[1]) + unit, name: r.name}); }
      else dots[i].setAttribute("visibility", "hidden");
    });
    const cb = c.getBoundingClientRect();
    tip.show(e.clientX - cb.left, e.clientY - cb.top, `${Math.round(near)} s`, rows);
  });
  hit.addEventListener("pointerleave", () => { tip.hide(); cross.setAttribute("visibility", "hidden");
    dots.forEach(d => d.setAttribute("visibility", "hidden")); });
  table(c, ["seconds", ...runs.map(r => r.name)],
    xs.map(v => [String(Math.round(v)), ...runs.map(r => { const p = r[key].find(q => q[0] === v); return p ? fmt(p[1]) : "–"; })]));
}

const num = v => v == null ? "–" : String(v);
const METRICS = [
  ["n", "problems", num], ["goodput", "correct and on time", pct], ["accuracy", "correct", pct],
  ["deadline_met", "on time", pct], ["goodput_baseline", "correct and on time, arrived below capacity", pct],
  ["goodput_burst", "correct and on time, arrived above capacity", pct], ["cancelled", "cancelled at the tail", num],
  ["latency_p50_s", "latency, median (s)", fmt], ["latency_p95_s", "latency, 95th percentile (s)", fmt],
  ["thinking_tokens_mean", "thinking tokens per workflow", fmt], ["think", "think calls", num],
  ["cut", "silent cuts", num], ["announced", "announced cuts", num], ["notices", "notices sent", num],
  ["p_mean", "mean pressure", v => v == null ? "–" : v.toFixed(3)], ["queue_frac", "time with a vLLM queue", pct],
  ["tpot_ms_mean", "decode time per token, mean (ms)", fmt]];
const GLOSSARY = [
  ["Correct and on time", "The answer matched MATH-500's and the whole workflow (plan, think, answer) finished before " +
   "its deadline. This is the headline number."],
  ["Deadline", "Arrival + slack x the problem's solo latency. Slack is 1.5, 3 or 6, drawn per problem."],
  ["No control", "vLLM runs every request with the full thinking budget it asked for."],
  ["Uniform cut", "Under pressure p, every running request's thinking budget drops to (1 - p) of what it asked for."],
  ["Brownout", "Like a uniform cut, but each cut starts from the request's predicted length and is weighted by " +
   "priority, so requests close to their deadline are cut less."],
  ["Silent, announced", "Silent stops the thought when the budget runs out. Announced writes a sentence into the " +
   "thought telling the model how many tokens it has left, so it can wrap up."],
  ["Classifier", "An online word model in the gateway that learns thinking lengths from finished requests."],
  ["Jev", "Open-Jev-2B, a small decision model on its own GPU, asked for each request's thinking length."],
  ["Pressure", "0 to 1 per vLLM replica: the queue (waiting / running), or with a decode SLO how far decoding is " +
   "past it ((time per token - SLO) / SLO), whichever is higher. Cuts start at 0.05."],
  ["Capacity C", "Workflows per second the GPUs finish when they are full; the trace's rates are multiples of it."],
  ["Spike", "The sharpest burst in the default BurstGPT window (Azure ChatGPT and GPT-4 traffic, KDD 25). Each " +
   "request becomes a MATH-500 problem, longer original answers getting harder levels."]];

function grid(c, head, rows, left = 1) {
  const t = el("table", {}, el("div", {class: "scroll"}, c));
  const tr = el("tr", {}, el("thead", {}, t));
  head.forEach((h, i) => el("th", i < left ? {text: h, class: "l"} : {text: h}, tr));
  const tb = el("tbody", {}, t);
  for (const r of rows) { const row = el("tr", {}, tb); r.forEach((v, i) => el("td", i < left ? {text: v, class: "l"} : {text: v}, row)); }
}

function facts(c, items) {
  const dl = el("dl", {class: "facts"}, c);
  for (const [k, v] of items) if (v != null && v !== "") { el("dt", {text: k}, dl); el("dd", {text: String(v)}, dl); }
}

function headline() {
  const runs = DATA.runs, none = runs.find(r => r.spec === "none");
  const best = runs.reduce((a, r) => (r.goodput || 0) > (a.goodput || 0) ? r : a, runs[0]);
  if (!best) return "";
  return none && best !== none
    ? `${best.name} kept ${pct(best.goodput)} of answers correct and on time, against ${pct(none.goodput)} with no control.`
    : `${best.name} kept ${pct(best.goodput)} of answers correct and on time.`;
}

function charts() {
  const runs = DATA.runs;
  const c = card("Answers that were correct and on time",
    "Every policy got the same arrivals and problems. On time means finished before the problem's deadline.");
  const tip = tipBox(c), multiples = el("div", {class: "multiples"}, c);
  bars(c, multiples, "Correct and on time", "goodput", tip);
  bars(c, multiples, "Correct", "accuracy", tip);
  bars(c, multiples, "On time", "deadline_met", tip);
  table(c, ["policy", "n", "correct and on time", "correct", "on time", "latency p50 (s)", "latency p95 (s)",
            "thinking tokens", "cuts", "notices", "cancelled"],
        runs.map(r => [r.name, String(r.n), pct(r.goodput), pct(r.accuracy), pct(r.deadline_met), fmt(r.latency_p50_s),
                       fmt(r.latency_p95_s), fmt(r.thinking_tokens_mean), num(r.cut), num(r.notices), num(r.cancelled)]));
  lines("Requests running in vLLM", "All vLLM replicas together, averaged over 5 s. Every running request shares the " +
        "GPU, so more of them means slower decoding for each.", "running", "");
  lines("Requests waiting in the vLLM queue", "All vLLM replicas together, averaged over 5 s.", "queue", "");
  if (runs.some(r => r.tpot.length))
    lines("Decode time per output token", "Milliseconds per token for each running request (running ÷ tokens " +
          "generated per second), averaged over replicas and 5 s." + (DATA.tpot_slo_ms ? " Above the SLO line, " +
          "brownout counts the slowdown as pressure." : ""), "tpot", " ms",
          DATA.tpot_slo_ms ? {value: DATA.tpot_slo_ms, label: `SLO ${DATA.tpot_slo_ms} ms`} : null);
  lines("Latency by arrival time", "Seconds, median over the finished workflows that arrived in each 20 s window " +
        "(cancelled ones are left out).", "latency", " s");
  lines("Thinking tokens by arrival time", "Tokens per finished workflow over all its think calls, median per 20 s " +
        "of arrivals; a cut shows up as fewer tokens.", "thinking", "");
  const more = el("p", {class: "foot"}, document.getElementById("app"));
  const a = el("a", {href: "#details", text: "Details: setup, statistics, every metric and a glossary"}, more);
  a.style.color = "var(--text-secondary)";
}

function details() {
  const runs = DATA.runs, S = DATA.setup || {};
  let c = card("Result");
  el("p", {class: "body", text: headline()}, c);
  if (DATA.note) el("p", {class: "body", text: DATA.note}, c);
  if (DATA.paired.length) {
    el("p", {class: "note", text: "Every policy ran the same problems at the same times, so each pair is compared " +
      "problem by problem. Gained: correct and on time with the second policy but not the first; lost: the reverse. " +
      "p is an exact McNemar test; below 0.05 is unlikely to be chance."}, c);
    grid(c, ["first", "second", "gained", "lost", "p"],
         DATA.paired.map(q => [q.a, q.b, String(q.gained), String(q.lost), q.p.toFixed(3)]), 2);
  }
  c = card("What was run");
  facts(c, [["Model", S.model], ["Problems", S.problems], ["Arrivals", S.arrivals], ["Load", S.load],
            ["Capacity", S.capacity], ["Deadlines", S.deadlines], ["Thinking budgets", S.budgets], ["Tail", S.tail],
            ["Brownout pressure", S.pressure], ["Length predictor", S.warmup], ["Seed", S.seed],
            ["Policies", runs.map(r => r.name).join("; ")], ["Suite", DATA.suite],
            ["Trace arguments", (DATA.trace_args || []).join(" ")]]);
  c = card("Every metric", "One column per policy.");
  grid(c, ["", ...runs.map(r => r.name)], METRICS.map(([k, name, f]) => [name, ...runs.map(r => f(r[k]))]));
  const bands = [...new Set(runs.flatMap(r => Object.keys(r.by_band || {})))].sort();
  if (bands.length) {
    c = card("Correct and on time by MATH level", "Share correct and on time, with the number of problems at that level.");
    grid(c, ["policy", ...bands], runs.map(r => [r.name, ...bands.map(b => {
      const x = (r.by_band || {})[b];
      return x ? `${pct(x.goodput)} (${x.n})` : "–";
    })]));
  }
  c = card("Glossary");
  facts(c, GLOSSARY);
}

const view = () => location.hash === "#details" ? "details" : "charts";

function render() {
  const app = document.getElementById("app");
  app.replaceChildren();
  const args = DATA.trace_args || [], wi = args.indexOf("--trace-window-h");
  const spike = DATA.suite === "spike" || (wi >= 0 && parseFloat(args[wi + 1]) <= 0.5);
  el("h1", {text: DATA.title || (spike ? "Thinking Brownout under one BurstGPT spike"
    : args.length ? "Thinking Brownout under a BurstGPT replay" : "Thinking Brownout under a rate profile")}, app);
  const nav = el("nav", {class: "tabs"}, app);
  for (const [id, name] of [["charts", "Charts"], ["details", "Details"]]) {
    const a = el("a", {href: "#" + id, text: name}, nav);
    if (view() === id) a.setAttribute("aria-current", "page");
  }
  if (view() === "details") details(); else charts();
}
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
addEventListener("hashchange", () => { render(); scrollTo(0, 0); });
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--title", help="page title and heading (default: from the suite)")
    ap.add_argument("--note", help="a line for the details page, e.g. the hardware")
    args = ap.parse_args()
    out = args.results / "report.html"
    data = {**collect(args.results), "title": args.title, "note": args.note}
    page = PAGE.replace("__TITLE__", html.escape(args.title or "Thinking Brownout report"))
    out.write_text(page.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    print(out)
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
