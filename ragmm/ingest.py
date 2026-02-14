from __future__ import annotations
from typing import Iterable, List
import os
import tempfile

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .settings import Settings


def split_docs(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def load_pdf_folder(folder: str) -> List[Document]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    loader = PyPDFDirectoryLoader(folder)
    docs = loader.load()
    # normalize metadata a bit
    for d in docs:
        d.metadata.setdefault("source_type", "kb_folder")
    return docs


def load_uploaded_pdfs(uploaded_files: Iterable) -> List[Document]:
    docs: List[Document] = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source_type"] = "upload"
            d.metadata["file_name"] = getattr(f, "name", "uploaded.pdf")
        docs.extend(file_docs)

    return docs