from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

from langchain_core.documents import Document


@dataclass
class EvalExample:
    question: str
    expected: Optional[str] = None        # substring expected in retrieved context (simple eval)
    gold_chunk_ids: Optional[List[str]] = None  # stronger eval if you have chunk_ids


def load_eval_examples_from_json(text: str) -> List[EvalExample]:
    """
    Accepts either:
    - JSON list: [{"question": "...", "expected": "..."}]
    - JSONL: one object per line
    """
    text = (text or "").strip()
    if not text:
        return []

    examples: List[EvalExample] = []

    if text.startswith("["):
        arr = json.loads(text)
        for obj in arr:
            examples.append(
                EvalExample(
                    question=obj["question"],
                    expected=obj.get("expected"),
                    gold_chunk_ids=obj.get("gold_chunk_ids"),
                )
            )
        return examples

    # JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        examples.append(
            EvalExample(
                question=obj["question"],
                expected=obj.get("expected"),
                gold_chunk_ids=obj.get("gold_chunk_ids"),
            )
        )

    return examples


def _doc_chunk_id(d: Document) -> str:
    return str((d.metadata or {}).get("chunk_id", ""))


def evaluate_retriever(
    retriever,
    examples: List[EvalExample],
    *,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Metrics:
    - hit@k: % of questions where at least one gold item is retrieved
      - if gold_chunk_ids present: check chunk_id match
      - else if expected string present: check substring in concatenated retrieved text
    - mrr: reciprocal rank for the first hit (0 if miss)
    """
    rows = []
    hits = 0
    mrr_sum = 0.0

    for ex in examples:
        docs = retriever.invoke(ex.question) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(ex.question)
        docs = docs[:k]

        hit = False
        rr = 0.0

        if ex.gold_chunk_ids:
            gold = set(ex.gold_chunk_ids)
            for rank, d in enumerate(docs, start=1):
                if _doc_chunk_id(d) in gold:
                    hit = True
                    rr = 1.0 / rank
                    break
        elif ex.expected:
            needle = ex.expected.strip().lower()
            for rank, d in enumerate(docs, start=1):
                if needle and needle in (d.page_content or "").lower():
                    hit = True
                    rr = 1.0 / rank
                    break
        else:
            # If user doesn't supply expected or gold, we can't score it
            hit = False
            rr = 0.0

        hits += 1 if hit else 0
        mrr_sum += rr

        rows.append(
            {
                "question": ex.question,
                "hit": hit,
                "rr": rr,
                "top_sources": [
                    (d.metadata or {}).get("source") or (d.metadata or {}).get("file_name") or "unknown"
                    for d in docs
                ],
            }
        )

    n = len(examples) if examples else 1
    return {
        "hit_at_k": hits / n,
        "mrr": mrr_sum / n,
        "rows": rows,
    }
