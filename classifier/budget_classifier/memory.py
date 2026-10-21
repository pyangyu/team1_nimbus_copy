from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .features import dot


@dataclass
class Example:
    x: dict[int, float]
    prompt: str
    thinking_tokens: int
    bucket: int
    model: str | None


class ExampleMemory:
    def __init__(self, size: int = 500, snippet_chars: int = 300):
        self.items: deque[Example] = deque(maxlen=size)
        self.snippet_chars = snippet_chars

    def add(self, x: dict[int, float], prompt: str, thinking_tokens: int, bucket: int, model: str | None) -> None:
        self.items.append(Example(x, prompt[: self.snippet_chars], thinking_tokens, bucket, model))

    def similar(self, x: dict[int, float], model: str | None, k: int = 8) -> list[Example]:
        pool = [e for e in self.items if e.model == model] or list(self.items)
        pool.sort(key=lambda e: dot(x, e.x), reverse=True)
        return pool[:k]
