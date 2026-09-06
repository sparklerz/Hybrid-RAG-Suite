from __future__ import annotations

import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from .settings import Settings

# Qwen3 bills thinking tokens as output tokens, which alone can exceed the
# per-model output-tokens-per-minute budget on Groq's free tier. "none" turns
# thinking off. Only qwen takes this param, so it is never sent to the
# gpt-oss/llama models.
QWEN_REASONING_EFFORT = os.getenv("QWEN_REASONING_EFFORT", "none")


def _reasoning_kwargs(model: str) -> dict:
    if "qwen" not in (model or "").lower():
        return {}
    return {"reasoning_effort": QWEN_REASONING_EFFORT}


def get_llm(settings: Settings, *, model: str | None = None, streaming: bool = True):
    if not settings.groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in environment.")

    name = model or settings.llm_model
    extra = _reasoning_kwargs(name)
    base = dict(
        groq_api_key=settings.groq_api_key,
        model_name=name,
        streaming=streaming,
    )
    try:
        return ChatGroq(**base, **extra)
    except (TypeError, ValueError):
        # Older langchain-groq has no reasoning_effort field; pass it straight
        # through to the API instead.
        return ChatGroq(**base, model_kwargs=extra)


def get_embeddings(settings: Settings):
    provider = settings.embed_provider.lower().strip()

    if provider == "huggingface":
        # HF_TOKEN is optional depending on environment/model
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    raise ValueError(f"Unsupported EMBED_PROVIDER: {settings.embed_provider}")