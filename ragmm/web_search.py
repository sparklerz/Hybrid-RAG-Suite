from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, wait
import os
import threading
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


# Wikimedia's User-Agent policy (https://w.wiki/4wJS) requires a descriptive UA
# identifying the app plus a contact. Browser-mimicking UAs share a heavily
# throttled bucket and get 429'd almost immediately; a missing UA gets 403'd.
_WIKI_CONTACT = os.getenv("WIKI_CONTACT", "https://github.com/hybrid-rag-suite")
_WIKI_HEADERS = {
    "User-Agent": os.getenv("WIKI_USER_AGENT", f"Hybrid-RAG-Suite/1.0 ({_WIKI_CONTACT}) python-requests"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip",
}

# Reuse connections across calls; also lets us keep one throttle/cache per process.
_WIKI_SESSION = requests.Session()
_WIKI_SESSION.headers.update(_WIKI_HEADERS)

# Small in-process TTL cache. Streamlit reruns the script on every interaction,
# so without this the same query hits the API repeatedly and burns the quota.
_WIKI_CACHE: Dict[str, tuple[float, str]] = {}
_WIKI_CACHE_TTL_S = 600.0
_WIKI_CACHE_MAX = 128
_WIKI_LOCK = threading.Lock()


# ddgs>=9 is a metasearch wrapper over several engines, not just DuckDuckGo.
# Its "auto" backend forces `wikipedia` and `grokipedia` to the front of the
# queue, and both currently return nothing, so a search can fail fast with
# "No results found." before a working engine is ever reached. Naming the
# engines that actually answer skips that dead prefix.
_DDG_BACKENDS = os.getenv("DDG_BACKENDS", "duckduckgo,brave,yahoo")

# Preferred web search. Set TAVILY_API_KEY to use it; ddgs is the fallback.
_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ---------- Tool implementations with hard timeouts ----------
def _ddg_search(query: str, *, max_results: int, timeout_s: float) -> str:
    # DDGS(timeout=...) bounds each *wave* of engines, not the whole search. ddgs
    # runs ceil(max_results/10)+1 engines at a time, so three backends take two
    # waves and the true ceiling is 2x timeout -- 12s for a 6s timeout, which
    # blows the caller's overall budget. Size the wave so both fit.
    deadline = time.monotonic() + timeout_s
    wave_timeout = max(2.0, timeout_s / 2)

    with DDGS(timeout=wave_timeout) as ddgs:
        try:
            hits = ddgs.text(query, max_results=max_results, backend=_DDG_BACKENDS)
        except Exception:
            # Either every named engine was rate-limited, or this is a pre-9.x
            # ddgs that rejects a comma-joined backend list. Retrying on its own
            # picks costs another full round, so only do it if the budget allows.
            if deadline - time.monotonic() < wave_timeout:
                raise
            hits = ddgs.text(query, max_results=max_results)

    rows = []
    for r in hits:
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        rows.append(f"- {title}\n  {href}\n  {body}".strip())
    return "\n\n".join(rows)


def _tavily_search(query: str, *, max_results: int, timeout_s: float) -> str:
    """Tavily REST API. Called directly rather than through langchain-tavily so
    the request stays inside this module's timeout budget."""
    r = requests.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": max_results, "search_depth": "basic"},
        headers={"Authorization": f"Bearer {_TAVILY_API_KEY}"},
        timeout=timeout_s,
    )
    r.raise_for_status()

    hits = r.json().get("results", []) or []
    if not hits:
        return "No results found."

    rows = []
    for h in hits[:max_results]:
        title = h.get("title", "")
        url = h.get("url", "")
        body = h.get("content", "")
        rows.append(f"- {title}\n  {url}\n  {body}".strip())
    return "\n\n".join(rows)


def _web_search(query: str, *, max_results: int, timeout_s: float) -> str:
    """Tavily when a key is configured, ddgs otherwise.

    ddgs scrapes consumer search engines, and those block datacenter egress, so
    it is unreliable on hosted platforms (HF Spaces, Streamlit Cloud, Colab)
    where the outbound IP is shared and cloud-owned. An API key authenticates by
    account rather than by source IP, so it behaves the same everywhere.
    """
    if not _TAVILY_API_KEY:
        return _ddg_search(query, max_results=max_results, timeout_s=timeout_s)

    deadline = time.monotonic() + timeout_s
    try:
        return _tavily_search(query, max_results=max_results, timeout_s=timeout_s)
    except Exception:
        # Out of quota or Tavily is down. ddgs may still work depending on where
        # this is deployed, but only try if there is budget left for it.
        remaining = deadline - time.monotonic()
        if remaining < 2.0:
            raise
        return _ddg_search(query, max_results=max_results, timeout_s=remaining)


def _wiki_summary(query: str, *, timeout_s: float) -> str:
    """
    Wikipedia via the MediaWiki API (NOT /wiki HTML, which 403s).

    One request does search + intro extract + canonical URL via `generator=search`,
    instead of the old search-then-REST-summary pair. Halving the request count
    matters because Wikimedia rate-limits per IP/UA and answers over-quota
    callers with an immediate 429.
    """
    key = query.strip().lower()
    now = time.monotonic()
    with _WIKI_LOCK:
        hit = _WIKI_CACHE.get(key)
        if hit and now - hit[0] < _WIKI_CACHE_TTL_S:
            return hit[1]

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exlimit": 1,
        "inprop": "url",
        "redirects": 1,
        "maxlag": 5,
    }

    deadline = now + timeout_s
    last_err: Optional[str] = None
    backoff = 0.5

    # Retry 429/503 within the caller's time budget instead of failing outright.
    for attempt in range(3):
        remaining = deadline - time.monotonic()
        if remaining <= 0.3:
            break
        r = _WIKI_SESSION.get(api, params=params, timeout=min(remaining, timeout_s))

        if r.status_code in (429, 503):
            last_err = f"{r.status_code} rate-limited by Wikimedia"
            retry_after = r.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else backoff
            except ValueError:
                delay = backoff
            backoff *= 2
            if time.monotonic() + delay >= deadline or attempt == 2:
                break
            time.sleep(delay)
            continue

        r.raise_for_status()
        js = r.json()

        # maxlag rejections come back as 200 with an error body
        if "error" in js:
            last_err = js["error"].get("info", str(js["error"]))
            break

        pages = js.get("query", {}).get("pages", []) or []
        if not pages:
            return "No Wikipedia results found."

        page = pages[0]
        title = page.get("title", "")
        url = page.get("fullurl", "")
        extract = (page.get("extract") or "").strip()
        text = f"""Title: {title}
URL: {url}

Summary:
{extract}""".strip()

        with _WIKI_LOCK:
            if len(_WIKI_CACHE) >= _WIKI_CACHE_MAX:
                _WIKI_CACHE.clear()
            _WIKI_CACHE[key] = (time.monotonic(), text)
        return text

    raise RuntimeError(last_err or "Wikipedia request failed")


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
        ("WebSearch", lambda: _web_search(query, max_results=ddg_k, timeout_s=per_tool_timeout_s)),
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