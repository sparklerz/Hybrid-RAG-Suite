from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import quote
import time
import requests
import xml.etree.ElementTree as ET

from ddgs import DDGS  # you already have ddgs in requirements
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from time import perf_counter


# ---------- Data structures ----------
@dataclass
class WebSearchResult:
    source: str
    content: str
    ok: bool = True
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None


def _truncate(text: str, max_chars: int = 2500) -> str:
    text = (text or "").strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


# Use a browser-like UA to avoid Wikimedia blocks
_WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAGDemo/1.0",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------- Tool implementations with hard timeouts ----------
def _ddg_search(query: str, *, max_results: int, timeout_s: float) -> str:
    rows = []
    with DDGS(timeout=timeout_s) as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            rows.append(f"- {title}\n  {href}\n  {body}".strip())
    return "\n\n".join(rows)


def _wiki_summary(query: str, *, timeout_s: float) -> str:
    """
    Wikipedia via API (NOT /wiki HTML), so it works even if /wiki is 403.
    1) Search best title via MediaWiki API
    2) Fetch summary via REST page/summary
    """
    api = "https://en.wikipedia.org/w/api.php"
    r = requests.get(
        api,
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=_WIKI_HEADERS,
        timeout=timeout_s,
    )
    r.raise_for_status()
    js = r.json()
    hits = js.get("query", {}).get("search", [])
    if not hits:
        return "No Wikipedia results found."

    title = hits[0]["title"]

    # REST summary
    rest = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    r2 = requests.get(rest, headers=_WIKI_HEADERS, timeout=timeout_s)
    r2.raise_for_status()
    js2 = r2.json()

    extract = js2.get("extract", "") or ""
    url = (js2.get("content_urls", {}) or {}).get("desktop", {}).get("page", "")
    return f"Title: {title}\nURL: {url}\n\nSummary:\n{extract}".strip()


def _arxiv_query(query: str, *, max_results: int, timeout_s: float) -> str:
    """
    arXiv Atom API, parsed with ElementTree (no extra deps).
    """
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)

    entries = root.findall("atom:entry", ns)
    if not entries:
        return "No arXiv results found."

    out = []
    for e in entries[:max_results]:
        title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        link = ""
        for ln in e.findall("atom:link", ns):
            if ln.attrib.get("rel") == "alternate":
                link = ln.attrib.get("href", "")
                break
        out.append(f"- {title}\n  {link}\n  {summary[:700]}{'...' if len(summary) > 700 else ''}")
    return "\n\n".join(out)


# # ---------- Heuristics: decide what to call ----------
# def _should_use_wiki(q: str) -> bool:
#     ql = q.lower()
#     return any(k in ql for k in ["who", "what", "age", "born", "biography", "capital", "president", "ceo", "founder"])


# def _should_use_arxiv(q: str) -> bool:
#     ql = q.lower()
#     return any(k in ql for k in ["paper", "arxiv", "study", "research", "preprint", "dataset", "benchmark", "llm"])


def _run_one(name: str, fn: Callable[[], str]) -> WebSearchResult:
    t0 = perf_counter()
    try:
        text = fn() or ""
        elapsed = int((perf_counter() - t0) * 1000)
        return WebSearchResult(name, _truncate(text), ok=True, elapsed_ms=elapsed)
    except Exception as e:
        elapsed = int((perf_counter() - t0) * 1000)
        return WebSearchResult(name, f"[{name} error] {e}", ok=False, error=str(e), elapsed_ms=elapsed)

# ---------- Main search function (bounded wall time) ----------
def search_web_wiki_arxiv(
    query: str,
    *,
    ddg_k: int = 5,
    arxiv_k: int = 3,
    per_tool_timeout_s: float = 6.0,
    overall_timeout_s: float = 7.0,
    enable_wiki: bool = True,
    enable_arxiv: bool = True,
) -> List[WebSearchResult]:
    """
    Runs DDG + Wikipedia(API) + arXiv concurrently (always, unless disabled).
    Hard-bounds total time and never blocks the UI indefinitely.
    """
    query = (query or "").strip()
    if not query:
        return [WebSearchResult("System", "Empty query.", ok=False, error="empty_query")]

    # Always do web
    tasks: List[tuple[str, Callable[[], str]]] = [
        ("WebSearch", lambda: _ddg_search(query, max_results=ddg_k, timeout_s=per_tool_timeout_s)),
    ]

    # Always do Wikipedia (API) if enabled
    if enable_wiki:
        tasks.append(("Wikipedia", lambda: _wiki_summary(query, timeout_s=per_tool_timeout_s)))

    # Always do arXiv if enabled
    if enable_arxiv:
        tasks.append(("Arxiv", lambda: _arxiv_query(query, max_results=arxiv_k, timeout_s=per_tool_timeout_s)))

    # results: List[WebSearchResult] = []
    # executor = ThreadPoolExecutor(max_workers=len(tasks))
    # future_map = {executor.submit(fn): name for name, fn in tasks}

    # done, not_done = wait(future_map.keys(), timeout=overall_timeout_s)

    # --- run tools in parallel (timeboxed) ---
    executor = ThreadPoolExecutor(max_workers=len(tasks))
    future_map = {executor.submit(_run_one, name, fn): name for name, fn in tasks}

    done, not_done = wait(future_map.keys(), timeout=overall_timeout_s)

    results: List[WebSearchResult] = []

    # Futures already return WebSearchResult
    for fut in done:
        try:
            results.append(fut.result())
        except Exception as e:
            name = future_map.get(fut, "UnknownTool")
            results.append(WebSearchResult(name, f"[{name} error] {e}", ok=False, error=str(e)))

    # Anything still running is treated as timeout
    for fut in not_done:
        name = future_map.get(fut, "UnknownTool")
        fut.cancel()
        results.append(
            WebSearchResult(
                name,
                f"[{name} timeout] exceeded overall_timeout_s={overall_timeout_s}s",
                ok=False,
                error="timeout",
            )
        )

    # Don't wait for stuck threads
    executor.shutdown(wait=False, cancel_futures=True)

    # keep stable order in UI
    order = {name: i for i, (name, _) in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r.source, 999))
    return results



# def search_web_wiki_arxiv(
#     query: str,
#     *,
#     ddg_k: int = 5,
#     arxiv_k: int = 3,
#     per_tool_timeout_s: float = 6.0,
#     overall_timeout_s: float = 7.0,
# ) -> List[WebSearchResult]:
#     """
#     Runs DDG + (maybe) Wiki + (maybe) arXiv concurrently.
#     Hard-bounds total time and never blocks the UI indefinitely.
#     """
#     query = (query or "").strip()
#     if not query:
#         return [WebSearchResult("System", "Empty query.", ok=False, error="empty_query")]

#     tasks: List[tuple[str, Callable[[], str]]] = []

#     # Always do web
#     tasks.append(("WebSearch", lambda: _ddg_search(query, max_results=ddg_k, timeout_s=per_tool_timeout_s)))

#     # Conditional wiki/arxiv
#     if _should_use_wiki(query):
#         tasks.append(("Wikipedia", lambda: _wiki_summary(query, timeout_s=per_tool_timeout_s)))
#     if _should_use_arxiv(query):
#         tasks.append(("Arxiv", lambda: _arxiv_query(query, max_results=arxiv_k, timeout_s=per_tool_timeout_s)))

#     results: List[WebSearchResult] = []
#     if not tasks:
#         return results

#     executor = ThreadPoolExecutor(max_workers=len(tasks))
#     future_map = {}

#     for name, fn in tasks:
#         future_map[executor.submit(fn)] = name

#     done, not_done = wait(future_map.keys(), timeout=overall_timeout_s)

#     # Collect done
#     for fut in done:
#         name = future_map[fut]
#         t0 = time.time()
#         try:
#             text = fut.result()
#             elapsed = int((time.time() - t0) * 1000)
#             results.append(WebSearchResult(name, _truncate(text), ok=True, elapsed_ms=elapsed))
#         except Exception as e:
#             elapsed = int((time.time() - t0) * 1000)
#             results.append(WebSearchResult(name, f"[{name} error] {e}", ok=False, error=str(e), elapsed_ms=elapsed))

#     # Handle timed out futures
#     for fut in not_done:
#         name = future_map[fut]
#         fut.cancel()
#         results.append(WebSearchResult(name, f"[{name} timeout] exceeded {overall_timeout_s}s", ok=False, error="timeout"))

#     # IMPORTANT: do not wait for stuck threads
#     executor.shutdown(wait=False, cancel_futures=True)

#     # Stable ordering for UI
#     order = {"WebSearch": 0, "Wikipedia": 1, "Arxiv": 2}
#     results.sort(key=lambda r: order.get(r.source, 99))

#     return results


# ---------- Answer function with fallback ----------
def answer_with_sources(llm, query: str, results: List[WebSearchResult]) -> Dict[str, object]:
    """
    If we have usable sources -> answer grounded ONLY in sources.
    If sources fail -> fallback to normal LLM answer (better UX), and label it.
    """
    good = [r for r in (results or []) if r.ok and r.content and not r.content.lower().startswith("[")]
    sources_blob = "\n\n".join([f"[{r.source}]\n{r.content}" for r in good])

    # If nothing usable, fallback to LLM (not grounded)
    if not good:
        fallback_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
            External search tools were unavailable or returned no reliable content.
            Answer the question using your general knowledge.
            Be transparent that you could not retrieve sources.

            Question:
            {question}
            """
        )
        chain = fallback_prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": query})
        return {"answer": answer, "sources": results, "used_fallback": True}

    prompt = ChatPromptTemplate.from_template(
        """You are a web research assistant.
        Answer the user question using ONLY the sources below.
        If the sources are insufficient, say what is missing.

        User question:
        {question}

        Sources:
        {sources}

        Instructions:
        - Write a concise, helpful answer.
        - Do NOT include citations or a "Sources used" line in your answer.
        """
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": query, "sources": sources_blob})
    return {"answer": answer, "sources": results, "used_fallback": False}