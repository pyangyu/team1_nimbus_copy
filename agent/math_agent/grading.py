from __future__ import annotations

import re

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MV = True
except Exception:
    HAVE_MV = False


def last_boxed(text: str) -> str | None:
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _norm(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("%", ""), (" ", "")):
        s = s.replace(a, b)
    return s.rstrip(".")


def _num(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract(text: str, dataset: str = "math") -> str | None:
    pred = last_boxed(text)
    if pred is None and dataset == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        pred = nums[-1] if nums else None
    return pred


def grade(pred: str | None, gold: str, dataset: str = "math") -> bool:
    if pred is None:
        return False
    if dataset == "gsm8k":
        a, b = _num(_norm(pred)), _num(gold)
        return a is not None and b is not None and abs(a - b) < 1e-6
    if _norm(pred) == _norm(gold):
        return True
    a, b = _num(_norm(pred)), _num(_norm(gold))
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    if HAVE_MV:
        try:
            return bool(mv_verify(mv_parse("$" + gold + "$"), mv_parse("$" + pred + "$")))
        except Exception:
            return False
    return False
