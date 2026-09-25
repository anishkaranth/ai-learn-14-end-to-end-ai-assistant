"""Deterministic tools: safe calculator, date arithmetic (fixed 'today'), and a KB lookup."""
from __future__ import annotations

import ast
import operator
import re
from datetime import date, timedelta
from typing import Dict, Optional

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 12:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.4f}".rstrip("0").rstrip(".")


def calculator(query: str) -> Optional[Dict]:
    q = query.lower().replace("×", "*").replace("^", "**")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", q)
    if m:
        expr = f"{m.group(1)}/100*{m.group(2)}"
    else:
        cands = re.findall(r"[\d\.\s\(\)\+\-\*/%]+", q)
        cands = [c.strip() for c in cands if re.search(r"\d", c) and re.search(r"[\+\-\*/%]", c)]
        if not cands:
            return None
        expr = max(cands, key=len)
    try:
        val = _eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return {"tool": "calculator", "input": expr, "output": None, "error": str(e)}
    return {"tool": "calculator", "input": expr, "output": _fmt(val)}


_ISO = r"(\d{4}-\d{2}-\d{2})"


def date_tool(query: str, today: date) -> Optional[Dict]:
    q = query.lower()
    d = [date.fromisoformat(s) for s in re.findall(_ISO, q)]
    m = re.search(r"(\d+)\s+days?\s+(after|from|before)\s+(today|" + _ISO + ")", q)
    if m:
        n, direction = int(m.group(1)), m.group(2)
        base = today if m.group(3) == "today" else date.fromisoformat(m.group(3))
        out = base + timedelta(days=n if direction != "before" else -n)
        return {"tool": "date", "input": f"{base} {'-' if direction == 'before' else '+'} {n}d", "output": out.isoformat()}
    if "between" in q and len(d) >= 2:
        return {"tool": "date", "input": f"{d[0]} .. {d[1]}", "output": str(abs((d[1] - d[0]).days))}
    if "day of the week" in q or "weekday" in q:
        target = d[0] if d else today
        return {"tool": "date", "input": f"weekday({target})", "output": target.strftime("%A")}
    if "today" in q and ("date" in q or "what day" in q):
        return {"tool": "date", "input": "today", "output": today.isoformat()}
    return None


def kb_lookup(query: str, kb: Dict) -> Optional[Dict]:
    m = re.search(r"ORD-\d{5}", query.upper())
    if m:
        oid = m.group(0)
        rec = kb["orders"].get(oid)
        return {"tool": "kb", "input": f"orders[{oid}]", "output": rec if rec else "not_found"}
    for plan in kb["plans"]:
        if re.search(rf"\b{plan}\b", query.lower()):
            return {"tool": "kb", "input": f"plans[{plan}]", "output": kb["plans"][plan]}
    return None
