#!/usr/bin/env python3
"""Ingest data/docs/*.md -> chunk -> TF-IDF -> artifacts/index.json."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from assistant.index import build_index, save_index

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "index.json"))
    ap.add_argument("--sentences-per-chunk", type=int, default=2)
    a = ap.parse_args()
    t0 = time.perf_counter()
    idx = build_index(Path(a.docs), a.sentences_per_chunk)
    save_index(idx, Path(a.out))
    print(f"indexed {idx['n_docs']} docs -> {idx['n_chunks']} chunks, vocab {idx['vocab_size']} "
          f"in {1000 * (time.perf_counter() - t0):.1f} ms -> {a.out}")


if __name__ == "__main__":
    main()
