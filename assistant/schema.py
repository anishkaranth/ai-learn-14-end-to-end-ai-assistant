"""From-scratch JSON-schema-like validator + the assistant's response schema."""
from __future__ import annotations

from typing import Any, Dict, List

_T = {"object": dict, "array": list, "string": str, "boolean": bool}


def validate(v: Any, s: Dict, path: str = "$") -> List[str]:
    t = s.get("type")
    if t == "number":
        if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            return [f"{path}: expected number"]
    elif t == "null_or_string":
        if not (v is None or isinstance(v, str)):
            return [f"{path}: expected string or null"]
    elif t and not isinstance(v, _T[t]):
        return [f"{path}: expected {t}"]
    errs = []
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in {s['enum']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v < s.get("minimum", float("-inf")) or v > s.get("maximum", float("inf")):
            errs.append(f"{path}: out of range")
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            errs += validate(x, s["items"], f"{path}[{i}]")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}: missing '{k}'")
        for k, x in v.items():
            if k in s.get("properties", {}):
                errs += validate(x, s["properties"][k], f"{path}.{k}")
            elif s.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected '{k}'")
    return errs


ROUTES = ["rag", "calculator", "date", "kb", "blocked", "abstain"]
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["query", "route", "answer", "citations", "tool_calls", "guardrails", "confidence", "latency_ms"],
    "properties": {
        "query": {"type": "string"},
        "route": {"type": "string", "enum": ROUTES},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "tool_calls": {"type": "array", "items": {"type": "object", "required": ["tool", "input"]}},
        "guardrails": {"type": "object", "required": ["input", "output"],
                       "properties": {"input": {"type": "string", "enum": ["allow", "redact", "block"]},
                                      "output": {"type": "string", "enum": ["allow", "redact", "block", "skipped"]}}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "latency_ms": {"type": "number", "minimum": 0},
    },
}
