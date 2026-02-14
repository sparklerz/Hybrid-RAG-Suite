from __future__ import annotations
import re
from typing import List, Tuple, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .settings import Settings


def sanitize_collection_name(name: str) -> str:
    """
    Chroma collection names should be simple.
    We'll keep letters, digits, underscores, hyphens. Replace others with underscore.
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "default"


def get_chroma(settings: Settings, embeddings, collection_name: str) -> Chroma:
    """
    One persistent Chroma DB at settings.chroma_dir, multiple collections inside.
    """
    name = sanitize_collection_name(collection_name)
    return Chroma(
        collection_name=name,
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )


def list_collections(settings: Settings) -> List[Tuple[str, int]]:
    """
    Returns (collection_name, doc_count).
    """
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    cols = client.list_collections()
    out: List[Tuple[str, int]] = []
    for c in cols:
        try:
            count = client.get_collection(c.name).count()
        except Exception:
            count = -1
        out.append((c.name, count))
    out.sort(key=lambda x: x[0])
    return out


def delete_collection(settings: Settings, name: str) -> None:
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    client.delete_collection(name)


def rebuild_collection(
    settings: Settings,
    embeddings,
    collection_name: str,
    docs: List[Document],
) -> int:
    """
    Deletes existing collection (if any), recreates, inserts docs.
    Returns number of chunks stored.
    """
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    name = sanitize_collection_name(collection_name)

    # Delete if exists
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)

    vs = get_chroma(settings, embeddings, name)
    vs.add_documents(docs)
    return len(docs)

def add_documents_to_collection(
    settings: Settings,
    embeddings,
    collection_name: str,
    docs: List[Document],
) -> int:
    """
    Appends docs into an existing collection (or creates it if missing).
    Returns number of chunks added.
    """
    name = sanitize_collection_name(collection_name)
    vs = get_chroma(settings, embeddings, name)
    vs.add_documents(docs)
    return len(docs)
