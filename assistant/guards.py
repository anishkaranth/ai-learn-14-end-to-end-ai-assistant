"""Input/output guardrails (regex-based, same design as ai-learn-13, trimmed for the assistant)."""
from __future__ import annotations

import re
from typing import Dict, List

INJECTION = [re.compile(p, re.I) for p in [
    r"\bignore\b.{0,30}\b(previous|prior|above|all|earlier)\b.{0,20}\b(instructions?|rules?|prompts?)",
    r"\bdisregard\b.{0,40}\b(instructions?|rules?|guidelines?|system)",
    r"\b(reveal|print|show|repeat|leak)\b.{0,30}\b(system|hidden|secret)\b.{0,15}\b(prompt|instructions?|message)",
    r"\byou are now\b|\bpretend (you are|to be)\b",
    r"\b(developer|dan|god) mode\b|\bjailbreak\b",
    r"</?(system|assistant)>",
]]
PII = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)\d(?:[ -]?\d){12,15}(?!\d)"),
}
CANARY = "CANARY-ACME-01"  # planted in the (notional) system prompt; must never be emitted
BANNED = ["idiot", "stupid", "hate you"]


def pii_kinds(text: str) -> List[str]:
    return [k for k, rx in PII.items() if rx.search(text)]


def redact(text: str) -> str:
    for k in ("credit_card", "email", "phone"):
        text = PII[k].sub(f"[{k.upper()}]", text)
    return text


def check_input(text: str) -> Dict:
    hits = [p.pattern[:32] for p in INJECTION if p.search(text)]
    kinds = pii_kinds(text)
    action = "block" if hits else ("redact" if kinds else "allow")
    return {"action": action, "pii": kinds, "text": text if action == "allow" else (redact(text) if action == "redact" else "")}


def check_output(text: str) -> Dict:
    reasons = (["system_prompt_leak"] if CANARY in text else []) + (["toxicity"] if any(b in text.lower() for b in BANNED) else [])
    kinds = pii_kinds(text)
    action = "block" if reasons else ("redact" if kinds else "allow")
    return {"action": action, "reasons": reasons,
            "text": "I can't share that." if action == "block" else (redact(text) if kinds else text)}
