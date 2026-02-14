from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Storage
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")

    # LLM
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "Llama3-8b-8192")

    # Embeddings
    embed_provider: str = os.getenv("EMBED_PROVIDER", "huggingface")  # huggingface|openai
    hf_token: str | None = os.getenv("HF_TOKEN")
    # openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # Defaults
    kb_pdf_dir: str = os.getenv("KB_PDF_DIR", "./kb_default")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))

    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "./data/artifacts")
