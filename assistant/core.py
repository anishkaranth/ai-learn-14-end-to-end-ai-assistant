"""The assistant: guard -> route -> (tool | RAG | abstain) -> guard -> validated JSON response."""
from __future__ import annotations

import re
import time
from datetime import date
from typing import Dict, List, Optional

from . import guards
from .index import search, split_sentences, tokenize
from .schema import RESPONSE_SCHEMA, validate
from .tools import calculator, date_tool, kb_lookup

DEFAULT_CONFIG = {"today": "2026-01-15", "top_k": 3, "abstain_threshold": 0.12, "max_answer_sentences": 2,
                  "min_relative_score": 0.5, "enable_tools": True}


class Assistant:
    def __init__(self, index: Dict, kb: Dict, config: Optional[Dict] = None):
        self.index, self.kb = index, kb
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.today = date.fromisoformat(self.cfg["today"])

    # ---------- routing ----------
    def route(self, text: str) -> str:
        if not self.cfg["enable_tools"]:
            return "rag"
        q = text.lower()
        if re.search(r"ord-\d{5}", q) or (re.search(r"\b(price|cost|how much)\b", q) and re.search(r"\b(free|pro|team)\b", q)):
            return "kb"
        if re.search(r"\d{4}-\d{2}-\d{2}", q) or re.search(r"\bdays? (after|from|before) today\b|\bday of the week\b|today'?s date", q):
            return "date"
        if re.search(r"\d\s*[\+\-\*/\^%]\s*[\d(]|\d+(\.\d+)?\s*%\s*of\s*\d", q):
            return "calculator"
        return "rag"

    # ---------- answerers ----------
    def _rag(self, text: str):
        hits = search(self.index, text, self.cfg["top_k"])
        top = float(hits[0][0]) if hits else 0.0
        if top < self.cfg["abstain_threshold"]:
            return "abstain", "I don't know based on the Acme Cloud help docs.", [], round(top, 4), hits
        qt = set(tokenize(text))
        idf = self.index["idf"]
        cands = []
        for rank, (score, ch) in enumerate(hits):
            if score < self.cfg["min_relative_score"] * top:  # ignore weakly related chunks
                continue
            for s in split_sentences(ch["text"]):
                overlap = sum(idf.get(t, 0.0) for t in set(tokenize(s)) & qt)
                cands.append((overlap + 0.5 * score - 0.01 * rank, s, ch["id"]))
        cands.sort(key=lambda x: -x[0])
        chosen = [c for c in cands if c[0] > 0][: self.cfg["max_answer_sentences"]] or cands[:1]
        answer = " ".join(c[1] for c in chosen)
        cites = list(dict.fromkeys(c[2] for c in chosen))
        return "rag", answer, cites, round(min(1.0, top), 4), hits

    def _tool(self, route: str, text: str):
        call = {"kb": lambda: kb_lookup(text, self.kb), "date": lambda: date_tool(text, self.today),
                "calculator": lambda: calculator(text)}[route]()
        if call is None or call.get("output") is None:
            return None
        out = call["output"]
        if route == "kb":
            if out == "not_found":
                answer = f"I couldn't find {call['input'].split('[')[1].rstrip(']')} in our records."
            elif "status" in out:
                eta = f", expected {out['eta']}" if out.get("eta") else ""
                answer = f"Order status: {out['status']} ({out['item']}{eta})."
            else:
                answer = f"The {call['input'][6:-1].title()} plan costs ${out['price_usd_month']} per month and includes {out['storage']} of storage."
        elif route == "date":
            answer = f"{out}"
        else:
            answer = f"{call['input']} = {out}"
        return answer, [call]

    # ---------- main entry ----------
    def answer(self, text: str) -> Dict:
        t0 = time.perf_counter()
        gin = guards.check_input(text)
        resp = {"query": text, "route": "blocked", "answer": "", "citations": [], "tool_calls": [],
                "guardrails": {"input": gin["action"], "output": "skipped"}, "confidence": 1.0}
        if gin["action"] == "block":
            resp["answer"] = "Request blocked by the input guardrail (possible prompt injection)."
        else:
            clean = gin["text"]
            route = self.route(clean)
            tool_res = self._tool(route, clean) if route != "rag" else None
            if tool_res is not None:
                resp.update(route=route, answer=tool_res[0], tool_calls=tool_res[1], confidence=1.0)
            else:  # RAG (also the fallback when a tool can't parse the request)
                r, ans, cites, conf, _ = self._rag(clean)
                resp.update(route=r, answer=ans, citations=cites, confidence=conf)
            gout = guards.check_output(resp["answer"])
            resp["answer"], resp["guardrails"]["output"] = gout["text"], gout["action"]
        resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        errs = validate(resp, RESPONSE_SCHEMA)
        if errs:  # never emit an invalid payload
            raise ValueError(f"response failed schema validation: {errs}")
        return resp
