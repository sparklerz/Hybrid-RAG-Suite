from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import math

from langchain_core.documents import Document
from langsmith import traceable


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


def _relevance(docs: List[Document], ex: EvalExample) -> Tuple[List[int], Optional[int]]:
    """
    Returns (rels, n_relevant):
    - rels[i] is 1 if the doc at rank i+1 is relevant, else 0
    - n_relevant is how many relevant items exist overall, or None when unknown
      (substring mode can't know how many chunks in the corpus contain the text)
    """
    if ex.gold_chunk_ids:
        gold = set(ex.gold_chunk_ids)
        seen = set()
        rels = []
        for d in docs:
            cid = _doc_chunk_id(d)
            # Count each gold chunk once, even if the retriever returns it twice
            if cid in gold and cid not in seen:
                seen.add(cid)
                rels.append(1)
            else:
                rels.append(0)
        return rels, len(gold)

    if ex.expected:
        needle = ex.expected.strip().lower()
        rels = [1 if needle and needle in (d.page_content or "").lower() else 0 for d in docs]
        return rels, None

    # If user doesn't supply expected or gold, we can't score it
    return [0] * len(docs), None


def _rank_metrics(rels: List[int], n_relevant: Optional[int], k: int) -> Dict[str, float]:
    found = sum(rels)
    hit = found > 0

    rr = 0.0
    for rank, rel in enumerate(rels, start=1):
        if rel:
            rr = 1.0 / rank
            break

    # Substring mode treats the expected text as a single target fact,
    # so recall collapses to "was it retrieved at all"
    recall = found / n_relevant if n_relevant else float(hit)
    precision = found / k

    # Best achievable number of relevant docs in the top k (denominator for AP and ideal DCG)
    ideal_n = min(n_relevant, k) if n_relevant else found

    ap = 0.0
    dcg = 0.0
    hits_so_far = 0
    for rank, rel in enumerate(rels, start=1):
        if rel:
            hits_so_far += 1
            ap += hits_so_far / rank
            dcg += 1.0 / math.log2(rank + 1)
    ap = ap / ideal_n if ideal_n else 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit": hit,
        "rr": rr,
        "recall": recall,
        "precision": precision,
        "ap": ap,
        "ndcg": ndcg,
    }


# One trace per evaluation run, with each question's retrieval nested under it.
# The retriever object itself is left out of the trace inputs.
@traceable(
    name="retriever_eval",
    process_inputs=lambda i: {"questions": [ex.question for ex in i["examples"]], "k": i.get("k")},
)
def evaluate_retriever(
    retriever,
    examples: List[EvalExample],
    *,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Relevance per retrieved doc:
      - if gold_chunk_ids present: chunk_id is in the gold set
      - else if expected string present: substring appears in the doc text

    Metrics (averaged over questions):
    - hit@k: % of questions where at least one relevant item is retrieved
    - mrr: reciprocal rank for the first hit (0 if miss)
    - recall@k: relevant retrieved / total relevant (substring mode: same as hit)
    - precision@k: relevant retrieved / k
    - map@k: average of precision@i at each relevant rank i, divided by min(total relevant, k)
    - ndcg@k: binary-relevance DCG divided by the DCG of an ideal ranking
    """
    rows = []
    totals = {"hit": 0.0, "rr": 0.0, "recall": 0.0, "precision": 0.0, "ap": 0.0, "ndcg": 0.0}

    for ex in examples:
        docs = retriever.invoke(ex.question) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(ex.question)
        docs = docs[:k]

        rels, n_relevant = _relevance(docs, ex)
        m = _rank_metrics(rels, n_relevant, k)

        for key in totals:
            totals[key] += float(m[key])

        rows.append(
            {
                "question": ex.question,
                **m,
                "relevance": rels,
                "top_sources": [
                    (d.metadata or {}).get("source") or (d.metadata or {}).get("file_name") or "unknown"
                    for d in docs
                ],
            }
        )

    n = len(examples) if examples else 1
    return {
        "hit_at_k": totals["hit"] / n,
        "mrr": totals["rr"] / n,
        "recall_at_k": totals["recall"] / n,
        "precision_at_k": totals["precision"] / n,
        "map_at_k": totals["ap"] / n,
        "ndcg_at_k": totals["ndcg"] / n,
        "rows": rows,
    }
