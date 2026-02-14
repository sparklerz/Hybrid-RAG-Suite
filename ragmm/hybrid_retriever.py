from __future__ import annotations
from typing import List, Dict
from langchain_core.documents import Document


def _get_chunk_id(d: Document) -> str:
    meta = d.metadata or {}
    return str(meta.get("chunk_id") or hash(d.page_content))


class HybridRetriever:
    """
    Combines a dense retriever + BM25 retriever using Reciprocal Rank Fusion (RRF).
    Both retrievers must provide invoke(query)->List[Document].
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        *,
        k: int = 5,
        dense_k: int = 8,
        bm25_k: int = 8,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_bm25: float = 1.0,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.k = k
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_bm25 = w_bm25

    def invoke(self, query: str) -> List[Document]:
        dense_docs = self.dense.invoke(query) if hasattr(self.dense, "invoke") else self.dense.get_relevant_documents(query)
        bm25_docs = self.bm25.invoke(query)

        dense_docs = dense_docs[: self.dense_k]
        bm25_docs = bm25_docs[: self.bm25_k]

        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # RRF for dense
        for rank, d in enumerate(dense_docs, start=1):
            cid = _get_chunk_id(d)
            doc_map[cid] = d
            scores[cid] = scores.get(cid, 0.0) + self.w_dense * (1.0 / (self.rrf_k + rank))

        # RRF for bm25
        for rank, d in enumerate(bm25_docs, start=1):
            cid = _get_chunk_id(d)
            doc_map[cid] = d
            scores[cid] = scores.get(cid, 0.0) + self.w_bm25 * (1.0 / (self.rrf_k + rank))

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[: self.k]

        out: List[Document] = []
        for cid in ranked_ids:
            d = doc_map[cid]
            meta = d.metadata or {}
            meta["chunk_id"] = str(meta.get("chunk_id") or cid)
            meta["retriever"] = meta.get("retriever", "hybrid")
            meta["hybrid_score"] = float(scores[cid])
            d.metadata = meta
            out.append(d)

        return out
