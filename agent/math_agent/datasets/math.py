from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import httpx

from . import Example

DATASETS = ("math500", "aime2026", "gsm8k")
URLS = {
    "math500": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
    "aime2026": "https://datasets-server.huggingface.co/rows?dataset=MathArena/aime_2026"
                "&config=default&split=train&offset=0&length=100",
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl",
}


def data_dir() -> Path:
    d = Path(os.environ.get("LIFELINE_DATA", Path(__file__).resolve().parents[2] / "data")) / "math"
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=None)
def _raw(name: str) -> list[dict]:
    path = data_dir() / f"{name}.jsonl"
    if not path.exists():
        r = httpx.get(URLS[name], follow_redirects=True, timeout=120)
        r.raise_for_status()
        if name == "aime2026":
            rows = [x["row"] for x in r.json()["rows"]]
            path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        else:
            path.write_bytes(r.content)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load(name: str = "math500", levels: list[int] | None = None) -> list[Example]:
    if name == "math500":
        items = [Example(x["unique_id"], x["problem"], x["answer"], "math", int(x["level"]))
                 for x in _raw(name)]
    elif name == "aime2026":
        items = [Example(f"aime_2026/{x['problem_idx']}", x["problem"], str(x["answer"]), "math")
                 for x in _raw(name)]
    elif name == "gsm8k":
        items = [Example(f"gsm8k/{i}", x["question"], x["answer"].split("####")[-1].strip(), "gsm8k")
                 for i, x in enumerate(_raw(name))]
    else:
        raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")
    if levels and name == "math500":
        items = [e for e in items if e.level in levels]
    return items
