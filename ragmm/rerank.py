from __future__ import annotations
from typing import List, Optional

from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.
    Works locally (CPU). Good for demo-quality improvements.

    Requires: sentence-transformers
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy import
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        # sort by score desc
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        out: List[Document] = []
        for i in order[:top_k]:
            d = docs[i]
            meta = d.metadata or {}
            meta["reranker"] = self.model_name
            meta["rerank_score"] = float(scores[i])
            d.metadata = meta
            out.append(d)
        return out


class RerankRetriever:
    """
    Wraps any retriever that supports .invoke(query)->List[Document]
    and reranks the returned documents.
    """

    def __init__(self, base_retriever, reranker: CrossEncoderReranker, *, fetch_k: int = 12, top_k: int = 6):
        self.base = base_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        docs = self.base.invoke(query) if hasattr(self.base, "invoke") else self.base.get_relevant_documents(query)
        docs = docs[: self.fetch_k]
        return self.reranker.rerank(query, docs, top_k=self.top_k)
