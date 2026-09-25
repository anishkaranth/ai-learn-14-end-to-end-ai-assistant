"""Document ingestion and a TF-IDF index that is saved to / loaded from a JSON artifact."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

STOP = set("a an the and or of to in on for with is are be by at it this that as from you your we our can any all "
           "do does how what when where which who why i my me if into out up after before than then there their "
           "was were will would should could has have had not no get gets".split())
_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    out = []
    for w in _TOK.findall(text.lower()):
        if w in STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # tiny stemmer: limits -> limit, invoices -> invoice
        out.append(w)
    return out


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def load_docs(doc_dir: Path) -> List[Dict]:
    docs = []
    for p in sorted(doc_dir.glob("*.md")):
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        docs.append({"doc_id": p.stem, "title": title, "text": " ".join(l.strip() for l in lines[1:] if l.strip())})
    return docs


def chunk_docs(docs: List[Dict], sentences_per_chunk: int = 2) -> List[Dict]:
    chunks = []
    for d in docs:
        sents = split_sentences(d["text"])
        for i in range(0, len(sents), sentences_per_chunk):
            chunks.append({"id": f"{d['doc_id']}#{i // sentences_per_chunk}", "doc_id": d["doc_id"], "title": d["title"],
                           "text": " ".join(sents[i:i + sentences_per_chunk])})
    return chunks


def _l2(vec: Dict[str, float], nd: int = 4) -> Dict[str, float]:
    n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: round(v / n, nd) for k, v in vec.items()}


def build_index(doc_dir: Path, sentences_per_chunk: int = 2) -> Dict:
    docs = load_docs(doc_dir)
    chunks = chunk_docs(docs, sentences_per_chunk)
    # the title is prepended so short chunks still carry their topic
    toks = [tokenize(c["title"] + " " + c["text"]) for c in chunks]
    df = Counter(t for ts in toks for t in set(ts))
    n = len(chunks)
    idf = {t: round(math.log((1 + n) / (1 + c)) + 1.0, 4) for t, c in sorted(df.items())}
    for c, ts in zip(chunks, toks):
        tf = Counter(ts)
        c["vec"] = _l2({t: (1 + math.log(f)) * idf[t] for t, f in sorted(tf.items())})
    return {"format": "tfidf-v1", "n_docs": len(docs), "n_chunks": n, "vocab_size": len(idf),
            "sentences_per_chunk": sentences_per_chunk, "idf": idf, "chunks": chunks}


def save_index(index: Dict, path: Path) -> None:
    """Compact but diff-friendly JSON: header fields, idf on one line, one chunk per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in index.items() if k not in ("idf", "chunks")}
    lines = ["{"] + [f"{json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append(f"\"idf\": {json.dumps(index['idf'], separators=(',', ':'))},")
    lines.append("\"chunks\": [")
    lines += [json.dumps(c, separators=(",", ":")) + ("," if i < len(index["chunks"]) - 1 else "")
              for i, c in enumerate(index["chunks"])]
    lines += ["]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_index(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def embed_query(index: Dict, text: str) -> Dict[str, float]:
    tf = Counter(t for t in tokenize(text) if t in index["idf"])
    return _l2({t: (1 + math.log(f)) * index["idf"][t] for t, f in tf.items()}, nd=6)


def search(index: Dict, query: str, k: int = 3) -> List[Tuple[float, Dict]]:
    q = embed_query(index, query)
    scored = [(sum(w * c["vec"].get(t, 0.0) for t, w in q.items()), c) for c in index["chunks"]]
    scored.sort(key=lambda x: -x[0])
    return [(round(s, 4), c) for s, c in scored[:k]]
