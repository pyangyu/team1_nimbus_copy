"""Executor for the FinQA program DSL.

A program is a comma-separated list of steps, e.g.
    subtract(5829, 5735), divide(#0, 5735)
`#i` refers to the result of step i. Constants are written `const_100`,
`const_m1` (= -1). Table ops take a row name and `none`:
    table_average(net revenue, none)
Semantics follow the official FinQA evaluation script.
"""
from __future__ import annotations

import math
import re

ARITH = {"add", "subtract", "multiply", "divide", "exp", "greater"}
TABLE = {"table_sum", "table_average", "table_max", "table_min"}
OPS = ARITH | TABLE

_STEP = re.compile(r"\s*([a-z_]+)\s*\(([^()]*)\)\s*,?")


class ProgramError(ValueError):
    pass


def parse_program(program: str) -> list[tuple[str, list[str]]]:
    program = program.strip().rstrip(".").strip()
    if program.upper().endswith("EOF"):
        program = program[:-3]
    steps, pos = [], 0
    while pos < len(program):
        m = _STEP.match(program, pos)
        if not m:
            if program[pos:].strip(" ,"):
                raise ProgramError(f"cannot parse at: {program[pos:pos + 40]!r}")
            break
        op, raw = m.group(1), m.group(2)
        if op not in OPS:
            raise ProgramError(f"unknown op {op!r}")
        if op in TABLE:
            # row names can contain commas, so split on the last one only
            name, _, _ = raw.rpartition(",")
            args = [name.strip() or raw.strip()]
        else:
            args = [a.strip() for a in raw.split(",")]
            if len(args) != 2:
                raise ProgramError(f"{op} takes 2 args, got {raw!r}")
        steps.append((op, args))
        pos = m.end()
    if not steps:
        raise ProgramError("empty program")
    return steps


def str_to_num(s: str) -> float:
    s = s.strip().replace(",", "").replace("$", "")
    if s.startswith("const_"):
        c = s[len("const_"):]
        return -1.0 if c == "m1" else float(c)
    if s.endswith("%"):
        return float(s[:-1]) / 100
    return float(s)


def cell_to_num(cell: str) -> float | None:
    # FinQA cells look like "$ 373", "-13 ( 13 )", "27.5% ( 27.5 % )", "( 4 )"
    s = cell.replace("$", "").replace(",", "").strip()
    if not s.startswith("(") and " (" in s:
        s = s.split(" (", 1)[0]
    s = s.replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    pct = s.endswith("%")
    try:
        v = float(s.rstrip("%"))
    except ValueError:
        return None
    v = -v if neg else v
    return v / 100 if pct else v


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def find_row(name: str, rows: dict[str, list[str]]) -> list[str]:
    key = _norm(name)
    norm = {_norm(k): v for k, v in rows.items()}
    if key in norm:
        return norm[key]
    hits = [v for k, v in norm.items() if key and (key in k or k in key)]
    if len(hits) == 1:
        return hits[0]
    raise ProgramError(f"table row not found: {name!r}")


def execute(program: str, rows: dict[str, list[str]] | None = None) -> float | str:
    """Run a program. `rows` maps a table row name to its value cells."""
    res: list[float | str] = []

    def arg(a: str) -> float:
        if a.startswith("#"):
            i = int(a[1:])
            if i >= len(res) or isinstance(res[i], str):
                raise ProgramError(f"bad step reference {a}")
            return res[i]  # type: ignore[return-value]
        try:
            return str_to_num(a)
        except ValueError:
            raise ProgramError(f"not a number: {a!r}") from None

    for op, args in parse_program(program):
        if op in TABLE:
            if rows is None:
                raise ProgramError("table op without a table")
            vals = [v for v in (cell_to_num(c) for c in find_row(args[0], rows)) if v is not None]
            if not vals:
                raise ProgramError(f"no numbers in row {args[0]!r}")
            r = {"table_sum": sum(vals), "table_average": sum(vals) / len(vals),
                 "table_max": max(vals), "table_min": min(vals)}[op]
        else:
            a, b = arg(args[0]), arg(args[1])
            if op == "add":
                r = a + b
            elif op == "subtract":
                r = a - b
            elif op == "multiply":
                r = a * b
            elif op == "divide":
                if b == 0:
                    raise ProgramError("division by zero")
                r = a / b
            elif op == "exp":
                r = a ** b
            else:
                r = "yes" if a > b else "no"
        if isinstance(r, float) and not math.isfinite(r):
            raise ProgramError("non-finite result")
        res.append(r)
    return res[-1]
