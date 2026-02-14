from __future__ import annotations
import json
import os
import pickle
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from .settings import Settings


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name.lower()).strip("_") or "default"


def docstore_path(settings: Settings, collection: str) -> str:
    os.makedirs(settings.artifacts_dir, exist_ok=True)
    d = os.path.join(settings.artifacts_dir, "docstore")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_safe_name(collection)}.jsonl")


def bm25_path(settings: Settings, collection: str) -> str:
    os.makedirs(settings.artifacts_dir, exist_ok=True)
    d = os.path.join(settings.artifacts_dir, "bm25")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_safe_name(collection)}.pkl")


def save_docstore(settings: Settings, collection: str, docs: List[Document]) -> None:
    path = docstore_path(settings, collection)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "page_content": d.page_content,
                "metadata": d.metadata or {},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_docstore(settings: Settings, collection: str) -> List[Document]:
    path = docstore_path(settings, collection)
    if not os.path.exists(path):
        return []
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec.get("metadata") or {}))
    return docs


def save_bm25_cache(settings: Settings, collection: str, chunk_ids: List[str], tokenized_corpus: List[List[str]]) -> None:
    path = bm25_path(settings, collection)
    with open(path, "wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "tokenized_corpus": tokenized_corpus}, f)


def load_bm25_cache(settings: Settings, collection: str) -> Tuple[List[str], List[List[str]]]:
    path = bm25_path(settings, collection)
    if not os.path.exists(path):
        return [], []
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj.get("chunk_ids", []), obj.get("tokenized_corpus", [])

def append_docstore(settings: Settings, collection: str, docs: List[Document]) -> None:
    """
    Append docs to the JSONL docstore for BM25/hybrid retrieval.
    """
    path = docstore_path(settings, collection)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    import json
    with open(path, "a", encoding="utf-8") as f:
        for d in docs:
            rec = {"page_content": d.page_content, "metadata": d.metadata or {}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def delete_bm25_cache(settings: Settings, collection: str) -> None:
    """
    Delete cached BM25 corpus so it is rebuilt next query.
    """
    path = bm25_path(settings, collection)
    if os.path.exists(path):
        os.remove(path)
