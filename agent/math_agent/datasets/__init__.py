"""Dataset interface. Each dataset yields Examples with a gold answer string;
grading.grade compares a predicted answer to it."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    id: str
    question: str
    gold: str
    dataset: str
    level: int = 0
