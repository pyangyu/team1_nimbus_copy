"""FinQA (Chen et al., EMNLP 2021): numeric QA over S&P 500 annual reports.

Each FinQA item is one report page (text + one table) and a question about it.
To make retrieval real work, the agent is only told which filing the question
is about (company/year) and must search every FinQA page of that filing,
pooled across train/dev/test.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import httpx

from . import Chunk, Example

URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
SPLITS = ("train", "dev", "test")


def data_dir() -> Path:
    d = Path(os.environ.get("LIFELINE_DATA", Path(__file__).resolve().parents[2] / "data")) / "finqa"
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=None)
def _raw(split: str) -> list[dict]:
    path = data_dir() / f"{split}.json"
    if not path.exists():
        r = httpx.get(URL.format(split=split), follow_redirects=True, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return json.loads(path.read_text(encoding="utf-8"))


def doc_of(item_id: str) -> str:
    # "ADI/2009/page_49.pdf-1" -> "ADI/2009"
    return item_id.rsplit("/", 1)[0]


def page_of(item_id: str) -> str:
    return item_id.rsplit("-", 1)[0]


def _gold(qa: dict) -> float | str:
    v = qa["exe_ans"]
    return v if isinstance(v, str) else float(v)


def load(split: str = "dev") -> list[Example]:
    return [Example(id=x["id"], question=x["qa"]["question"], gold=_gold(x["qa"]),
                    doc=doc_of(x["id"]), gold_program=x["qa"]["program"])
            for x in _raw(split)]


def page_rows(table: list[list[str]]) -> dict[str, list[str]]:
    return {r[0]: r[1:] for r in table[1:] if r and r[0].strip()}


def _row_text(header: list[str], row: list[str]) -> str:
    parts = [f"{h.strip() or 'value'} = {c}" for h, c in zip(header[1:], row[1:]) if c.strip()]
    return f"{row[0]}: " + "; ".join(parts)


@lru_cache(maxsize=1)
def corpus() -> dict[str, list[Chunk]]:
    """Chunks per filing: one per sentence and one per table row, deduped by page."""
    out: dict[str, list[Chunk]] = defaultdict(list)
    seen: set[str] = set()
    for split in SPLITS:
        for x in _raw(split):
            page = page_of(x["id"])
            if page in seen:
                continue
            seen.add(page)
            doc = doc_of(x["id"])
            for i, s in enumerate(x["pre_text"] + x["post_text"]):
                if len(s.strip()) > 3:
                    out[doc].append(Chunk(f"{page}:text_{i}", doc, s.strip()))
            table = x["table"]
            for i, row in enumerate(table[1:], start=1):  # FinQA numbers rows incl. header
                if row and any(c.strip() for c in row[1:]):
                    out[doc].append(Chunk(f"{page}:table_{i}", doc, _row_text(table[0], row),
                                          row_name=row[0], cells=row[1:]))
    return dict(out)


def gold_rows(item_id: str, split: str = "dev") -> dict[str, list[str]]:
    for x in _raw(split):
        if x["id"] == item_id:
            return page_rows(x["table"])
    raise KeyError(item_id)
