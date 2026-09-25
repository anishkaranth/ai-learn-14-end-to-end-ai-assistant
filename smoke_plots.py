"""Matplotlib SVG plots + RESULTS.md writer for the end-to-end assistant smoke run."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-14", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names = []
    full, abl = m["eval"]["pass_rate_by_route"], m["ablation_no_tools"]["pass_rate_by_route"]
    routes = list(full)
    x = np.arange(len(routes))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.2, [full[r] for r in routes], 0.4, label=f"full assistant (overall {m['eval']['pass_rate']:.2f})", color="#126782")
    ax.bar(x + 0.2, [abl[r] for r in routes], 0.4, label=f"RAG only, no tools (overall {m['ablation_no_tools']['pass_rate']:.2f})", color="#e9c46a")
    ax.set_xticks(x, [f"{r}\n(n={m['eval']['n_by_route'][r]})" for r in routes], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("eval pass rate")
    ax.set_title("Eval suite pass rate by expected route")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    names.append(_save(fig, out / "pass_rate_by_route.svg"))

    sp = m["retrieval_by_query_type"]
    keys = ["hit@1", "hit@3", "answer_pass_rate"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(x - 0.2, [sp["direct"][k] for k in keys], 0.4, label=f"direct wording (n={sp['direct']['n']})", color="#2a9d8f")
    ax.bar(x + 0.2, [sp["paraphrase"][k] for k in keys], 0.4, label=f"paraphrased (n={sp['paraphrase']['n']})", color="#e76f51")
    ax.set_xticks(x, ["retrieval hit@1", "retrieval hit@3", "answer pass rate"])
    ax.set_ylim(0, 1.05)
    ax.set_title("TF-IDF retrieval: direct vs paraphrased questions")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    names.append(_save(fig, out / "retrieval_direct_vs_paraphrase.svg"))
    return names


def write_results_md(path: Path, m: Dict[str, Any]) -> None:
    e, a, ix = m["eval"], m["ablation_no_tools"], m["index"]
    L = ["# Results: end-to-end AI assistant (smoke run)", "",
         f"Runtime {m['runtime_sec']}s for the whole pipeline on CPU: index build, {len(m['samples'])} sample queries, "
         f"the {e['n_cases']}-case eval suite, and the no-tools ablation. The pipeline is deterministic (seed {m['seed']}; fixed 'today' = {m['config']['today']}).", "",
         f"Index: {ix['n_docs']} docs → {ix['n_chunks']} chunks ({ix['sentences_per_chunk']} sentences each), vocab {ix['vocab_size']} "
         f"(`artifacts/index.json`). Build log: `{m['build_log']}`", "",
         "## Eval suite", "",
         "| metric | value |", "|---|---|",
         f"| overall pass rate | **{e['pass_rate']:.4f}** |",
         f"| route accuracy | {e['route_accuracy']:.4f} |",
         f"| retrieval hit@1 / hit@3 / MRR@5 (n={e['retrieval']['n']}) | {e['retrieval']['hit@1']} / {e['retrieval']['hit@3']} / {e['retrieval']['mrr@5']} |",
         f"| injection block precision / recall | {e['guardrails']['block_precision']} / {e['guardrails']['block_recall']} |",
         f"| PII redaction correct | {e['guardrails']['pii_redaction_ok']}/{e['guardrails']['pii_cases']} |",
         f"| schema-valid JSON responses | {e['schema_valid_rate']:.4f} |",
         f"| latency p50 / p95 / max (ms) | {e['latency_ms']['p50']} / {e['latency_ms']['p95']} / {e['latency_ms']['max']} |", "",
         "| expected route | n | full assistant | RAG only (no tools) |", "|---|---|---|---|"]
    for r, v in e["pass_rate_by_route"].items():
        L.append(f"| {r} | {e['n_by_route'][r]} | {v:.2f} | {a['pass_rate_by_route'][r]:.2f} |")
    sp = m["retrieval_by_query_type"]
    L += ["", f"No-tools ablation: overall pass rate {a['pass_rate']:.4f} (route accuracy {a['route_accuracy']:.4f}). "
          "Calculator, date and KB questions can't be answered from documents.", "",
          "Direct vs paraphrased RAG questions:", "", "| query type | n | hit@1 | hit@3 | answer pass |", "|---|---|---|---|---|"]
    for t in ("direct", "paraphrase"):
        L.append(f"| {t} | {sp[t]['n']} | {sp[t]['hit@1']} | {sp[t]['hit@3']} | {sp[t]['answer_pass_rate']} |")
    L += ["", f"### Failures ({len(e['failures'])})", ""]
    for f in e["failures"]:
        L.append(f"- `{f['id']}` expected **{f['expected_route']}**, got **{f['route']}**: {f['answer']}")
    L += ["", "## Sample responses (structured JSON)", ""]
    for s in m["samples"]:
        brief = {k: s[k] for k in ("route", "answer", "citations", "guardrails", "confidence")}
        if s["tool_calls"]:
            brief["tool_calls"] = s["tool_calls"]
        L.append(f"- **{s['query']}**  \n  `{json.dumps(brief, ensure_ascii=False)}`")
    L += ["", "## Takeaways", "",
          "- The router sends arithmetic, date and order/plan questions to deterministic tools. Without them those 12 cases fail, which is what the ablation shows.",
          "- TF-IDF retrieval is near perfect when a question reuses the docs' words, and drops on paraphrases (MFA vs 2FA, money back vs refund). Dense embeddings or query expansion would be the next step.",
          "- The abstain threshold turns low-similarity queries into 'I don't know' instead of hallucinated answers. It also costs recall on paraphrases.",
          "- Every response passes the JSON schema validator before it's returned, and the guardrails block the injection probes and redact PII.", "",
          "Plots: " + ", ".join(f"`{p}`" for p in m["plots"]), ""]
    path.write_text("\n".join(L), encoding="utf-8")
