"""Final-answer extraction for the probe labeler.

Kept local so the gateway does not depend on the agent package; the rules mirror math_agent.grading.
"""
from __future__ import annotations

import re

EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution "
              "based on the thinking directly now.\n")
FINAL_PREFIX = "The final answer is \\boxed{"


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


def normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("%", ""), (" ", ""), (",", "")):
        s = s.replace(a, b)
    return s.rstrip(".")


def final_answer(text: str) -> str | None:
    """The boxed answer in text, normalized; falls back to the last number; None if there is nothing."""
    pred = last_boxed(text)
    if pred is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        pred = nums[-1] if nums else None
    return None if pred is None else normalize(pred)
