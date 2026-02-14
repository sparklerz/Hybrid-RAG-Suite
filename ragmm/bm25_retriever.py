from __future__ import annotations
import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from .settings import Settings
from .artifacts import load_docstore, load_bm25_cache, save_bm25_cache


_token_re = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())


class BM25Retriever:
    def __init__(self, settings: Settings, collection: str, *, k: int = 5):
        self.settings = settings
        self.collection = collection
        self.k = k

        self._docs: List[Document] = []
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: List[str] = []

        self._load_or_build()

    def _load_or_build(self):
        docs = load_docstore(self.settings, self.collection)
        self._docs = docs

        chunk_ids, tokenized = load_bm25_cache(self.settings, self.collection)

        # If cache missing or out-of-date, rebuild
        if not tokenized or len(tokenized) != len(docs):
            tokenized = [_tokenize(d.page_content) for d in docs]
            chunk_ids = [(d.metadata or {}).get("chunk_id", str(i)) for i, d in enumerate(docs)]
            save_bm25_cache(self.settings, self.collection, chunk_ids, tokenized)

        self._chunk_ids = chunk_ids
        self._bm25 = BM25Okapi(tokenized)

    def invoke(self, query: str) -> List[Document]:
        if not self._bm25 or not self._docs:
            return []

        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        # top indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]

        out: List[Document] = []
        for i in ranked:
            d = self._docs[i]
            meta = d.metadata or {}
            meta["chunk_id"] = str(meta.get("chunk_id") or self._chunk_ids[i])
            meta["retriever"] = "bm25"
            meta["bm25_score"] = float(scores[i])
            d.metadata = meta
            out.append(d)
        return out
