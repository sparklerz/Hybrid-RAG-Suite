from __future__ import annotations
import os
import shutil
from typing import List, Tuple

from .settings import Settings
from .vectorstore import list_collections, sanitize_collection_name
from .artifacts import docstore_path, bm25_path


def available_collections(settings: Settings) -> List[str]:
    cols = [name for name, _ in list_collections(settings)]
    return cols


def collection_stats(settings: Settings) -> List[Tuple[str, int, bool, bool]]:
    """
    Returns: (collection_name, chunk_count, has_docstore, has_bm25_cache)
    """
    out = []
    for name, count in list_collections(settings):
        has_docstore = os.path.exists(docstore_path(settings, name))
        has_bm25 = os.path.exists(bm25_path(settings, name))
        out.append((name, count, has_docstore, has_bm25))
    return out


def delete_collection_artifacts(settings: Settings, collection: str) -> None:
    ds = docstore_path(settings, collection)
    bm = bm25_path(settings, collection)
    for p in (ds, bm):
        if os.path.exists(p):
            os.remove(p)
