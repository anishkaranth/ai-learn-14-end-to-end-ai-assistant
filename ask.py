#!/usr/bin/env python3
"""Ask the assistant: `python ask.py "question"` for one JSON answer, or `python ask.py` for a REPL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assistant.core import Assistant
from assistant.index import load_index

ROOT = Path(__file__).resolve().parent


def load_assistant(index_path: Path, **cfg) -> Assistant:
    if not index_path.exists():
        sys.exit(f"index not found at {index_path}; run `python build_index.py` first")
    kb = json.loads((ROOT / "data" / "kb.json").read_text(encoding="utf-8"))
    return Assistant(load_index(index_path), kb, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="*", help="question (omit for interactive mode)")
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "index.json"))
    a = ap.parse_args()
    bot = load_assistant(Path(a.index))
    if a.query:
        print(json.dumps(bot.answer(" ".join(a.query)), indent=2))
        return
    print("Acme Cloud assistant. Type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print(json.dumps(bot.answer(q), indent=2))


if __name__ == "__main__":
    main()
