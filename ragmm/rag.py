from __future__ import annotations
from typing import List
from typing import Callable, Dict, Any, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
import re
import requests

KURAL_LINE_RE = re.compile(r"(?m)^\s*(\d{1,4})\s*[.)]?\s*$")
KURAL_TOKEN_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
KURAL_HEAD_RE = re.compile(r"(?m)^\s*(\d{1,4})\s*[.)]\s+")  # e.g. "15.  'Tis rain works..."
KURAL_WORD_RE = re.compile(r"(?i)\b(?:kural|couplet)\s*#?\s*(\d{1,4})\b")  # e.g. "Kural 103:"

def _extract_kural_nums(text: str, max_n: int = 3) -> List[int]:
    text = text or ""
    nums: List[int] = []

    # Prefer "15. <text>" style (your English KB PDF format)
    for m in KURAL_HEAD_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= 1330:
            nums.append(n)
    # Also accept "Kural 103:" / "Couplet 220" style
    for m in KURAL_WORD_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= 1330:
            nums.append(n)

    out = []
    for n in nums:
        if n not in out:
            out.append(n)
    return out[:max_n]

def _fetch_kural_tamil_from_api(n: int, timeout_s: float = 4.0) -> str:
    """
    Fallback Tamil source:
    https://tamil-kural-api.vercel.app/api/kural/<n>

    Returns Tamil couplet as 2 lines, or "" if unavailable.
    """
    if not (1 <= n <= 1330):
        return ""
    url = f"https://tamil-kural-api.vercel.app/api/kural/{n}"
    try:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        js = r.json() or {}
        k = js.get("kural")
        if isinstance(k, list):
            txt = "\n".join([str(x).strip() for x in k if str(x).strip()])
            return txt.strip()
        return ""
    except Exception:
        return ""

def _build_tamil_map_from_nums(nums: List[int]) -> str:
    """
    Build a labeled map so the LLM can print Tamil for MULTIPLE kurals.
    Format:
    [15]
    line1
    line2

    [220]
    line1
    line2
    """
    out: List[str] = []
    for n in (nums or []):
        api_tamil = _fetch_kural_tamil_from_api(n)
        if _looks_like_tamil(api_tamil):
            out.append(f"[{n}]\n{api_tamil}".strip())
    return "\n\n".join(out).strip()

def _looks_like_tamil(s: str) -> bool:
    return sum("\u0B80" <= ch <= "\u0BFF" for ch in (s or "")) >= 12

## If multiple relevant couplets exist, return up to 5.

THIRUKKURAL_STYLE = """You are answering questions about Thirukkural.

Hard rules:
- Explanation must be in English.
- You MUST choose only Kural numbers that appear in <kural_numbers>.
- For EACH chosen Kural number N, you MUST print the Tamil couplet from <tamil_map> under label [N].
- Print Tamil in a markdown code block and it MUST be EXACTLY 2 lines.
- You MUST copy-paste the two lines exactly as-is from <tamil_map> (no rewriting).
- Do NOT invent or reconstruct Tamil text.
- If a chosen N is missing from <tamil_map>, say Tamil couplet not available for that N.
- If you cannot find a relevant Tamil couplet in context, say so.

Output format (repeat this block per couplet; separate blocks with a blank line and '---'):

Kural number: <N>

Kural (Tamil):
```text
<copy the two-line Tamil exactly from <tamil_map> label [N]>
```

Translation (English):
<use translation from context if present; otherwise translate faithfully>

Explanation (English):
<2-6 sentences grounded in the context>
"""


def _resolve_style(answer_style: Optional[str]) -> str:
    if not answer_style:
        return ""
    s = answer_style.strip().lower()
    if s in {"thirukkural", "kural", "thirukural"}:
        return THIRUKKURAL_STYLE
    # allow passing a custom instruction string directly
    return answer_style.strip()


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _retrieve_docs(retriever, query: str):
    """
    Version-safe retriever call.
    Newer LangChain retrievers (VectorStoreRetriever) are Runnables -> use .invoke(query).
    Older retrievers may have .get_relevant_documents(query).
    """
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    raise AttributeError("Retriever has neither .invoke nor .get_relevant_documents")


def make_basic_rag(llm, retriever, *, answer_style: str | None = None):
    """
    Basic RAG: retrieve -> stuff context -> answer.
    Returns a runnable whose invoke({"input": ...}) returns {"answer": str, "context": docs}
    """
    style = _resolve_style(answer_style)
    system = "Answer using ONLY the provided context."
    if style:
        system += "\n\n" + style

    prompt = ChatPromptTemplate.from_template(
        f"""{system}

<context>
{{context}}
</context>

Question: {{input}}
"""
    )

    parser = StrOutputParser()

    def retrieve(inputs: Dict[str, Any]):
        return _retrieve_docs(retriever, inputs["input"])

    chain = (
        RunnablePassthrough.assign(**{"context_docs": retrieve})
        | RunnablePassthrough.assign(context=lambda x: _format_docs(x["context_docs"]))
        | {"answer": prompt | llm | parser, "context": lambda x: x["context_docs"]}
    )
    return chain


def make_conversational_rag(
    llm,
    retriever,
    get_history: Callable[[str], BaseChatMessageHistory],
    *,
    answer_style: str | None = None,
):
    """
    Conversational RAG with question rewrite using chat history.
    Returns a RunnableWithMessageHistory. Invoke returns {"answer": str, "context": docs}
    """
    style = _resolve_style(answer_style)

    # 1) Rewrite question using history
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the user question as a standalone question. Do NOT answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    parser = StrOutputParser()
    rewrite_chain = rewrite_prompt | llm | parser

    # 2) Answer using retrieved docs
    system = (
        "You are a helpful assistant. Use retrieved context to answer. "
        "If unsure, say you don't know. Keep it concise."
    )
    if style:
        system += "\n\n" + style

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system
                + "\n\n<kural_numbers>\n{kural_numbers}\n</kural_numbers>"
                + "\n\n<tamil_map>\n{tamil_map}\n</tamil_map>"
                + "\n\n<context>\n{context}\n</context>",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    def retrieve_from_question(inputs: Dict[str, Any]):
        return _retrieve_docs(retriever, inputs["input"])
    core = (
        RunnablePassthrough.assign(
            input=lambda x: rewrite_chain.invoke(
                {"input": x["input"], "chat_history": x.get("chat_history", [])}
            )
        )
        | RunnablePassthrough.assign(**{"context_docs": retrieve_from_question})
        | RunnablePassthrough.assign(context=lambda x: _format_docs(x["context_docs"]))
        # | RunnablePassthrough.assign(tamil_snippets=lambda x: _build_tamil_snippets(x, style, retriever))
        | RunnablePassthrough.assign(kural_numbers=lambda x: _extract_kural_nums(x.get("context",""), max_n=12))
        | RunnablePassthrough.assign(tamil_map=lambda x: _build_tamil_map_from_nums(x["kural_numbers"]))
        | {"answer": qa_prompt | llm | parser, "context": lambda x: x["context_docs"]}
    )

    return RunnableWithMessageHistory(
        core,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def make_history_store() -> dict[str, ChatMessageHistory]:
    return {}


def get_or_create_history(store: dict[str, ChatMessageHistory], session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
