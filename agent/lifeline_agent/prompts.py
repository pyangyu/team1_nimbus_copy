from __future__ import annotations

from .datasets import Chunk

SYSTEM = "You are a financial analyst answering questions about a company's annual report."

DSL_RULES = """Write the calculation as a program in this language:
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)  greater(a, b)
  table_sum(row name, none)  table_average(row name, none)  table_max(row name, none)  table_min(row name, none)
Arguments are numbers copied from the evidence (no units, no $, no commas), #i for the result of step i (0-based), or constants like const_100, const_1000, const_m1.
Steps are separated by ", " and cannot be nested. A percent change is a fraction: subtract(new, old), divide(#0, old).
Examples:
  program: subtract(5829, 5735), divide(#0, 5735)
  program: add(120, 85), divide(#0, const_2)
  program: table_average(net revenue, none)"""


def plan_messages(question: str, doc: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"Filing: {doc}\nQuestion: {question}\n\n"
             "List the specific facts you need to find in the report to answer this "
             "(line items, years, units). One per line, at most 5 lines, no other text."}]


def evidence_block(chunks: list[Chunk]) -> str:
    return "\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))


def reason_messages(question: str, doc: str, chunks: list[Chunk]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"Filing: {doc}\nEvidence from the report:\n{evidence_block(chunks)}\n\n"
             f"Question: {question}\n\n{DSL_RULES}\n\n"
             "Reply with exactly one line starting with \"program:\"."}]


def check_messages(question: str, chunks: list[Chunk], program: str, result) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"Evidence from the report:\n{evidence_block(chunks)}\n\n"
             f"Question: {question}\nProposed calculation: {program}\nResult: {result}\n\n"
             "Does the evidence contain the right numbers, and does this calculation answer the "
             "question? Reply with only a confidence between 0 and 1."}]
