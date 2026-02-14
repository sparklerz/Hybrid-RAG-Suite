from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from .settings import Settings


def get_llm(settings: Settings, *, model: str | None = None, streaming: bool = True):
    if not settings.groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in environment.")
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=model or settings.llm_model,
        streaming=streaming,
    )


def get_embeddings(settings: Settings):
    provider = settings.embed_provider.lower().strip()

    if provider == "huggingface":
        # HF_TOKEN is optional depending on environment/model
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    raise ValueError(f"Unsupported EMBED_PROVIDER: {settings.embed_provider}")