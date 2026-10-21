from __future__ import annotations

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def plan_messages(question: str) -> list[dict]:
    return [{"role": "user", "content":
             f"Problem:\n{question}\n\n"
             "Do not solve it. In at most 4 short lines, list the approach you would take "
             "(key facts, methods or cases). No other text."}]


def solve_messages(question: str, plan: str, rejected: list[str]) -> list[dict]:
    notes = f"\n\nApproach notes (may be incomplete):\n{plan.strip()}" if plan.strip() else ""
    if rejected:
        notes += ("\n\nEarlier attempts answered " + ", ".join(rejected)
                  + ", which a check judged likely wrong. Re-derive the answer carefully.")
    return [{"role": "user", "content": question + notes + INSTR}]


def verify_messages(question: str, answer: str) -> list[dict]:
    return [{"role": "user", "content":
             f"Problem:\n{question}\n\nProposed final answer: {answer}\n\n"
             "Is this answer correct? Reply with only a confidence between 0 and 1."}]
