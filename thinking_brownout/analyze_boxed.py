#!/usr/bin/env python3
r"""How much thinking comes AFTER the model has already written its answer down?

Models often write \boxed{...} inside their thinking when they have an answer, and
then keep going. This script uses that as a free signal and asks, for two cut rules:
how many tokens would the cut save, and would the answer still be the same?

  rule "first"   cut the first time ANY \boxed{value} appears in the thinking
  rule "twice"   cut the second time the SAME \boxed{value} appears (the model has
                 confirmed its answer once)

No model is run. For a trace where the rule fires, we assume a cut there would make
the model answer with the boxed value it had just written. That is an assumption;
only a real cut run can confirm it.

  python analyze_boxed.py traces/*.jsonl
  python analyze_boxed.py traces/*.jsonl --min-tokens 8192     # only the long traces
"""
import json
import re
import sys

import numpy as np

SKIP = {"", "answer", "...", "x", "?", "final answer", "your answer"}   # placeholders, not answers


def norm(s):
    s = str(s).strip().rstrip(".")
    s = s.replace("\\displaystyle", "")
    s = re.sub(r"^(-?[\d.,]+)\s*\\(?:text|mbox|mathrm)\{\s*[a-zA-Z ]+\}$", r"\1", s.strip())   # 480\text{ dollars}
    s = re.sub(r"\\(?:text|mbox|mathrm)\{([^}]*)\}", r"\1", s)
    for a, b in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\$", ""), ("$", ""), ("^\\circ", ""), ("\\%", ""),
                 ("%", ""), (" ", "")]:
        s = s.replace(a, b)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):          # 1,000 -> 1000 (not "1,-1" or "1,2")
        s = s.replace(",", "")
    return s


def _fix_frac(s):
    r"""\frac12 -> \frac{1}{2}, \frac1{16} -> \frac{1}{16}, \frac{7}2 -> \frac{7}{2}"""
    def arg(t, i):
        if i < len(t) and t[i] == "{":
            d, j = 1, i + 1
            while j < len(t) and d:
                d += {"{": 1, "}": -1}.get(t[j], 0)
                j += 1
            return t[i:j], j
        return ("{" + t[i] + "}", i + 1) if i < len(t) else ("{}", i)
    out, i = [], 0
    while True:
        j = s.find("\\frac", i)
        if j < 0:
            return "".join(out) + s[i:]
        a, k = arg(s, j + 5)
        b, k = arg(s, k)
        out.append(s[i:j] + "\\frac" + a + b)
        i = k


def _to_expr(s):
    r"""LaTeX number -> sympy-parsable text, or None if it is not a plain math expression."""
    if len(s) > 80:
        return None
    t = s
    for _ in range(4):                                             # inside out: roots, then fractions
        t = re.sub(r"\\sqrt\[(\d+)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t)
        t = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", t)
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    t = t.replace("{", "(").replace("}", ")")
    if "\\" in t or not re.fullmatch(r"[0-9a-z+\-*/(). ]*", t) or not t.strip():
        return None
    if re.search(r"[a-z]", re.sub(r"sqrt|pi", "", t)):             # free variables: not a number
        return None
    if re.search(r"\*\*\s*\(?\d{4,}", t):                          # huge powers: do not evaluate
        return None
    t = re.sub(r"(\d|\))\s*(sqrt|pi|\()", r"\1*\2", t)              # 2sqrt(3), 3pi, 2(…) -> explicit *
    t = re.sub(r"(pi)\s*(\d|sqrt|\()", r"\1*\2", t)
    return t


def _value(s):
    e = _to_expr(s)
    if e is None:
        return None
    try:
        import sympy
        v = complex(sympy.sympify(e, rational=True).evalf(30))
        return v if abs(v.imag) < 1e-12 else None
    except Exception:
        return None


def _split_top(s):
    """'[a,b]' -> ('[', ']', ['a','b']); 'a,b' -> ('', '', ['a','b'])."""
    o = c = ""
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        depth, ok = 0, True
        for i, ch in enumerate(s):
            depth += 1 if ch in "([{" else -1 if ch in ")]}" else 0
            if depth == 0 and i < len(s) - 1:
                ok = False
                break
        if ok:
            o, c, s = s[0], s[-1], s[1:-1]
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += 1 if ch in "([{" else -1 if ch in ")]}" else 0
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return o, c, parts


def _same_one(a, b):
    if a == b:
        return True
    va, vb = _value(a), _value(b)
    if va is not None and vb is not None:
        return abs(va - vb) <= 1e-6 * max(1.0, abs(vb))
    return False


def same(a, b):
    r"""Equivalence of two final answers: same text after normalising, or the same number
    (\frac12 = \frac{1}{2} = 0.5 = 1/2, 28917/98901 = 119/407, 2\sqrt3 = \sqrt{12}),
    element by element for tuples / intervals / lists (lists without brackets in any order)."""
    a, b = norm(a), norm(b)
    if a == b:
        return True
    a, b = _fix_frac(a), _fix_frac(b)
    a = re.sub(r"^[a-zA-Z]\w*=", "", a)                            # "x=5" -> "5"
    b = re.sub(r"^[a-zA-Z]\w*=", "", b)
    if a == b:
        return True
    oa, ca, pa = _split_top(a)
    ob, cb, pb = _split_top(b)
    if len(pa) != len(pb) or (len(pa) > 1 and (oa, ca) != (ob, cb)):
        return False
    if len(pa) == 1:
        return _same_one(pa[0], pb[0])
    if oa:                                                          # tuple / interval: in order
        return all(_same_one(x, y) for x, y in zip(pa, pb))
    left = list(pb)                                                 # bare list: any order
    for x in pa:
        hit = next((y for y in left if _same_one(x, y)), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def boxed_all(text):
    """Every \\boxed{...} in the text as (end position, value), in order."""
    out, i = [], 0
    while True:
        i = text.find("\\boxed{", i)
        if i < 0:
            return out
        j, depth = i + 7, 1
        while j < len(text) and depth > 0:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            val = text[i + 7:j - 1]
            if norm(val).lower() not in SKIP:
                out.append((j, val))
        i = j


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return [r for r in rows if r.get("think_tokens")]


def cut_point(boxes, rule):
    """(position, value) where the rule fires, or None."""
    if rule == "first":
        return boxes[0] if boxes else None
    seen = []
    for pos, val in boxes:
        if any(same(val, v) for v in seen):
            return pos, val
        seen.append(val)
    return None


def report(path, min_tokens=0):
    rows = [r for r in read(path) if r["think_tokens"] >= min_tokens]
    if not rows:
        print(f"\n=== {path}: no traces with at least {min_tokens} thinking tokens ===")
        return
    total = sum(r["think_tokens"] for r in rows)
    acc_now = np.mean([bool(r.get("correct")) for r in rows])
    if min_tokens:
        n_len = sum(r.get("finish_reason") == "length" for r in rows)
        n_ok = sum(bool(r.get("correct")) for r in rows)
        print(f"\n--- only traces with >= {min_tokens} thinking tokens: {n_len / len(rows):.0%} hit the cap, "
              f"{n_ok / len(rows):.0%} correct, {1 - (n_len + n_ok) / len(rows):.0%} finished but wrong ---")
    print(f"\n=== {path}: {len(rows)} traces, {total / 1e6:.1f}M thinking tokens, accuracy now {acc_now:.3f} ===")
    print(f"{'rule':<8}{'fires on':>10}{'cut at':>9}{'tokens saved':>14}{'same answer':>13}"
          f"{'answer changed':>16}{'accuracy after':>16}")
    for rule in ("first", "twice"):
        fired = saved = same_ans = fired_fin = 0
        pos_frac, correct_after = [], 0
        for r in rows:
            ok_now = bool(r.get("correct"))
            cp = cut_point(boxed_all(r["thinking"]), rule) if r.get("thinking") else None
            if cp is None:
                correct_after += ok_now
                continue
            pos, val = cp
            frac = pos / max(len(r["thinking"]), 1)
            fired += 1
            pos_frac.append(frac)
            saved += r["think_tokens"] * (1 - frac)
            final = boxed_all(r.get("answer") or "")
            if r.get("finished") and final:        # compare only where a final answer exists
                fired_fin += 1
                same_ans += same(val, final[-1][1])
            correct_after += same(val, r["gold"])
        n = len(rows)
        print(f"{rule:<8}{fired / n:>10.0%}{np.median(pos_frac) if pos_frac else float('nan'):>9.0%}"
              f"{saved / total:>14.0%}{same_ans / max(fired_fin, 1):>13.0%}"
              f"{1 - same_ans / max(fired_fin, 1):>16.0%}{correct_after / n:>16.3f}")
    trunc = [r for r in rows if r.get("finish_reason") == "length"]
    if trunc:
        had = sum(bool(boxed_all(r["thinking"])) for r in trunc)
        right = sum(any(same(v, r["gold"]) for _, v in boxed_all(r["thinking"])) for r in trunc)
        print(f"traces that hit the cap: {len(trunc)}; {had} had written a boxed answer in the thinking, "
              f"{right} of those had written the correct one")
    print("fires on        share of traces where the rule triggers")
    print("cut at          median position of the cut inside the thinking")
    print("tokens saved    share of ALL thinking tokens after the cut point")
    print("same answer     of the finished traces where it fires: the boxed value equals the final answer")
    print("answer changed  the model later changed its mind (so the extra thinking mattered)")
    print("accuracy after  estimated accuracy if every fired trace answered with the boxed value")


if __name__ == "__main__":
    args = sys.argv[1:]
    min_tokens = 0
    if "--min-tokens" in args:                 # look only at long traces, e.g. --min-tokens 8192
        i = args.index("--min-tokens")
        min_tokens = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    for p in args:
        report(p, min_tokens)
