"""Dataset interface. Each dataset yields Examples plus a per-document corpus
the agent retrieves from. FinQA is first; TAT-QA plugs in the same way."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    doc: str  # retrieval scope, e.g. "ADI/2009" for one company filing
    text: str
    row_name: str | None = None  # set for table rows
    cells: list[str] = field(default_factory=list)


@dataclass
class Example:
    id: str
    question: str
    gold: float | str
    doc: str
    gold_program: str = ""


def score(pred: float | str | None, gold: float | str) -> tuple[bool, bool]:
    """(strict, lenient). Strict is FinQA execution accuracy. Lenient also
    accepts 1% relative error and percent-vs-fraction scaling, since the
    question text often says "percent" while exe_ans is a fraction."""
    if pred is None:
        return False, False
    if isinstance(gold, str) or isinstance(pred, str):
        ok = str(pred).strip().lower() == str(gold).strip().lower()
        return ok, ok
    strict = abs(pred - gold) <= max(1e-5, 1e-4 * abs(gold))

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(1e-3, 0.01 * abs(b))

    lenient = strict or close(pred, gold) or close(pred, gold * 100) or close(pred * 100, gold)
    return strict, lenient
