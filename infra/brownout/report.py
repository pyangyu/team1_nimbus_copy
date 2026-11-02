"""Chart a spike or realistic suite run: python3 infra/brownout/report.py brownout-results [--open]

Writes report.html into the results folder: correct, on time and correct-and-on-time per policy, the vLLM queue
over the run, and latency and thinking tokens by arrival time.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
NAMES = {"none": "No control", "uniform": "Uniform cut", "brownout": "Brownout", "reject": "Reject", "defer": "Defer"}
QUEUE_BIN_S = 5.0
ARRIVAL_BIN_S = 20.0


def label(spec: str) -> str:
    spec, _, model = spec.partition("@")
    policy, notice = (spec.split(":") + [""])[:2]
    out = NAMES.get(policy, policy)
    if policy in ("uniform", "brownout"):
        out += ", announced" if notice == "announced" else ", silent"
    return out + (f" + {model}" if model else "")


def run_dir(root: Path, spec: str) -> Path:
    spec, _, model = spec.partition("@")
    return root / f"realistic-{spec.replace(':', '-')}{'-' + model if model else ''}"


def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def vllm(d: Path, t0: float, col: int) -> list[list[float]]:
    by: dict[tuple[int, int], list[float]] = {}
    for row in jsonl(d / "pressure.jsonl"):
        by.setdefault((int((row[0] - t0) // QUEUE_BIN_S), row[4] if len(row) > 4 else 0), []).append(row[col])
    bins: dict[int, float] = {}
    for (k, _), xs in by.items():
        bins[k] = bins.get(k, 0.0) + statistics.fmean(xs)
    return [[round(k * QUEUE_BIN_S, 1), round(v, 2)] for k, v in sorted(bins.items()) if k >= 0]


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


def collect(root: Path) -> dict:
    res = json.loads((root / "results.json").read_text())
    runs, profile = [], ""
    for k, r in enumerate(res["runs"]):
        d = run_dir(root, r["policy"])
        s = json.loads((d / "summary.json").read_text())
        wfs = jsonl(d / "workflows.jsonl")
        profile = profile or r.get("profile") or ""
        runs.append({"spec": r["policy"], "name": label(r["policy"]), "light": LIGHT[k % 8], "dark": DARK[k % 8],
                     "n": r["n"], "accuracy": r["accuracy"], "deadline_met": r["deadline_met"],
                     "goodput": r["correct_and_on_time"], "cancelled": r.get("cancelled"),
                     "latency_p50_s": r.get("latency_p50_s"), "latency_p95_s": r.get("latency_p95_s"),
                     "thinking_tokens_mean": r.get("thinking_tokens_mean"), "cut": r.get("cut"),
                     "notices": r.get("notices"), "queue": vllm(d, s["t0"], 2), "running": vllm(d, s["t0"], 3),
                     "latency": by_arrival(wfs, s["t0"], "latency_s"),
                     "thinking": by_arrival(wfs, s["t0"], "thinking_tokens")})
    cap = res.get("capacity") or {}
    warm = res.get("warmup") or {}
    return {"suite": res.get("suite"), "workload": res.get("workload"), "trace_args": res.get("trace_args"),
            "capacity": cap.get("capacity_wf_s"), "warmup": warm, "bursts": bursts(profile), "profile": profile,
            "runs": runs}


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thinking Brownout report</title>
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
  const runs = DATA.runs, W = 320, rowH = 30, left = 132, right = 44, top = 6, H = top + runs.length * rowH + 22;
  const svg = el("svg:svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title}, box);
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

function lines(title, note, key, unit) {
  const c = card(title, note), tip = tipBox(c), runs = DATA.runs;
  const lg = el("div", {class: "legend"}, c);
  for (const r of runs) { const s = el("span", {}, lg); const k = el("span", {class: "key"}, s);
    k.style.background = color(r); s.appendChild(document.createTextNode(r.name)); }
  const W = 900, H = 280, L = 52, R = 16, T = 10, B = 34;
  const xs = [...new Set(runs.flatMap(r => r[key].map(p => p[0])))].sort((a, b) => a - b);
  const xmax = Math.max(60, ...xs), ymax = niceMax(Math.max(0, ...runs.flatMap(r => r[key].map(p => p[1]))));
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

function render() {
  const app = document.getElementById("app");
  app.replaceChildren();
  const runs = DATA.runs;
  const args = DATA.trace_args || [], wi = args.indexOf("--trace-window-h");
  const spike = DATA.suite === "spike" || (wi >= 0 && parseFloat(args[wi + 1]) <= 0.5);
  el("h1", {text: spike ? "Thinking Brownout under one BurstGPT spike"
    : args.length ? "Thinking Brownout under a BurstGPT replay" : "Thinking Brownout under a rate profile"}, app);
  const n = runs.length ? runs[0].n : 0;
  const sub = [`${n} MATH-500 problems per policy, the same problems at the same times in every run`];
  if (DATA.capacity) sub.push(`capacity C = ${DATA.capacity} workflows/s`);
  if (DATA.warmup && DATA.warmup.labels) sub.push(`length predictor warmed on ${DATA.warmup.n} other problems`);
  el("p", {class: "sub", text: sub.join(" · ")}, app);
  const none = runs.find(r => r.spec === "none");
  const best = runs.reduce((a, r) => (r.goodput || 0) > (a.goodput || 0) ? r : a, runs[0]);
  if (best) el("p", {class: "headline", text: none && best !== none
    ? `${best.name} kept ${pct(best.goodput)} of answers correct and on time, against ${pct(none.goodput)} with no control.`
    : `${best.name} kept ${pct(best.goodput)} of answers correct and on time.`}, app);
  const c = card("Answers that were correct and on time",
    "Every policy got the same arrivals and problems. On time means finished before the problem's deadline.");
  const tip = tipBox(c), grid = el("div", {class: "multiples"}, c);
  bars(c, grid, "Correct and on time", "goodput", tip);
  bars(c, grid, "Correct", "accuracy", tip);
  bars(c, grid, "On time", "deadline_met", tip);
  table(c, ["policy", "n", "correct and on time", "correct", "on time", "latency p50 (s)", "latency p95 (s)",
            "thinking tokens", "cuts", "notices", "cancelled"],
        runs.map(r => [r.name, String(r.n), pct(r.goodput), pct(r.accuracy), pct(r.deadline_met), fmt(r.latency_p50_s),
                       fmt(r.latency_p95_s), fmt(r.thinking_tokens_mean), String(r.cut ?? "–"), String(r.notices ?? "–"),
                       String(r.cancelled ?? "–")]));
  lines("Requests running in vLLM", "Both replicas together, averaged over 5 s. Every running request shares the " +
        "GPU, so more of them means slower decoding for each.", "running", "");
  lines("Requests waiting in the vLLM queue", "Both replicas together, averaged over 5 s. Brownout's pressure is " +
        "waiting ÷ running on each replica.", "queue", "");
  lines("Latency by arrival time", "Seconds, median over the finished workflows that arrived in each 20 s window " +
        "(cancelled ones are left out).", "latency", " s");
  lines("Thinking tokens by arrival time", "Tokens per finished workflow over all its think calls, median per 20 s " +
        "of arrivals; a cut shows up as fewer tokens.", "thinking", "");
  el("p", {class: "foot", text: `Suite ${DATA.suite} · workload ${DATA.workload} · arrivals ${(DATA.trace_args || []).join(" ")} · profile ${DATA.profile}`}, app);
}
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    out = args.results / "report.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(collect(args.results)).replace("</", "<\\/")))
    print(out)
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
