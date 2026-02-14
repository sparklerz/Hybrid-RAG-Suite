from __future__ import annotations
import hashlib
from typing import List
from langchain_core.documents import Document


def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def ensure_chunk_ids(docs: List[Document]) -> List[Document]:
    """
    Ensures each Document has metadata['chunk_id'] (stable hash).
    """
    for d in docs:
        meta = d.metadata or {}
        source = str(meta.get("source") or meta.get("file_name") or meta.get("source_type") or "unknown")
        page = str(meta.get("page", meta.get("page_number", "")))
        key = f"{source}::{page}::{d.page_content}"
        meta["chunk_id"] = meta.get("chunk_id") or _stable_hash(key)
        d.metadata = meta
    return docs
