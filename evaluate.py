#!/usr/bin/env python3
"""Run the eval suite (data/eval_set.json) against the assistant; print a summary and optionally save JSON."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ask import load_assistant
from assistant.index import search
from assistant.schema import RESPONSE_SCHEMA, validate

ROOT = Path(__file__).resolve().parent


def grade(case: Dict, resp: Dict) -> bool:
    if resp["route"] != case["route"]:
        return False
    if case["route"] == "rag":
        return all(k.lower() in resp["answer"].lower() for k in case["keywords"])
    if case["route"] in ("calculator", "date", "kb"):
        return case["expected"].lower() in resp["answer"].lower()
    return True  # blocked / abstain: the route is the whole answer


def evaluate(bot, cases: List[Dict]) -> Dict:
    rows, by_route = [], defaultdict(list)
    ranks = []
    for c in cases:
        r = bot.answer(c["query"])
        ok = grade(c, r)
        row = {"id": c["id"], "expected_route": c["route"], "route": r["route"], "pass": ok, "answer": r["answer"],
               "citations": r["citations"], "latency_ms": r["latency_ms"], "schema_valid": not validate(r, RESPONSE_SCHEMA),
               "input_guard": r["guardrails"]["input"]}
        if "gold_doc" in c:  # retrieval quality independent of routing
            docs = [ch["doc_id"] for _, ch in search(bot.index, c["query"], 5)]
            rank = docs.index(c["gold_doc"]) + 1 if c["gold_doc"] in docs else None
            ranks.append(rank)
            row["gold_rank"] = rank
        if "expect_input_guard" in c:
            row["input_guard_ok"] = r["guardrails"]["input"] == c["expect_input_guard"]
        rows.append(row)
        by_route[c["route"]].append(ok)
    lat = np.array([r["latency_ms"] for r in rows])
    blocked_true = [c["route"] == "blocked" for c in cases]
    blocked_pred = [r["route"] == "blocked" for r in rows]
    tp = sum(a and b for a, b in zip(blocked_true, blocked_pred))
    fp = sum((not a) and b for a, b in zip(blocked_true, blocked_pred))
    fn = sum(a and not b for a, b in zip(blocked_true, blocked_pred))
    return {
        "n_cases": len(cases),
        "pass_rate": round(float(np.mean([r["pass"] for r in rows])), 4),
        "route_accuracy": round(float(np.mean([r["route"] == r["expected_route"] for r in rows])), 4),
        "pass_rate_by_route": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_route.items())},
        "n_by_route": {k: len(v) for k, v in sorted(by_route.items())},
        "retrieval": {"n": len(ranks), "hit@1": round(float(np.mean([r == 1 for r in ranks])), 4),
                      "hit@3": round(float(np.mean([r is not None and r <= 3 for r in ranks])), 4),
                      "mrr@5": round(float(np.mean([1 / r if r else 0.0 for r in ranks])), 4)},
        "guardrails": {"block_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                       "block_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                       "pii_redaction_ok": sum(r.get("input_guard_ok", False) for r in rows),
                       "pii_cases": sum("input_guard_ok" in r for r in rows)},
        "schema_valid_rate": round(float(np.mean([r["schema_valid"] for r in rows])), 4),
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 3), "p95": round(float(np.percentile(lat, 95)), 3),
                       "max": round(float(lat.max()), 3)},
        "failures": [{k: r[k] for k in ("id", "expected_route", "route", "answer")} for r in rows if not r["pass"]],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "index.json"))
    ap.add_argument("--cases", default=str(ROOT / "data" / "eval_set.json"))
    ap.add_argument("--no-tools", action="store_true", help="ablation: route everything to RAG")
    ap.add_argument("--out", default=None, help="optional path to write the full report JSON")
    a = ap.parse_args()
    bot = load_assistant(Path(a.index), enable_tools=not a.no_tools)
    rep = evaluate(bot, json.loads(Path(a.cases).read_text(encoding="utf-8")))
    summary = {k: v for k, v in rep.items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
