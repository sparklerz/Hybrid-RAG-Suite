from __future__ import annotations
from typing import Literal, List
import re

import validators
import requests
from bs4 import BeautifulSoup

from youtube_transcript_api import YouTubeTranscriptApi
import youtube_transcript_api as yta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def detect_url_type(url: str) -> Literal["youtube", "web", "invalid"]:
    if not url or not validators.url(url):
        return "invalid"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "web"


def _extract_youtube_id(url: str) -> str | None:
    # supports:
    # - https://www.youtube.com/watch?v=VIDEOID
    # - https://youtu.be/VIDEOID
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{6,})", url)
    return m.group(1) if m else None


def _fetch_youtube_transcript(url: str, preferred_langs: List[str] | None = None) -> str:
    vid = _extract_youtube_id(url)
    if not vid:
        raise ValueError("Could not extract YouTube video ID from the URL.")

    preferred_langs = preferred_langs or ["en", "en-US", "en-GB"]

    # Option B: try alternate YouTube hosts by patching youtube_transcript_api settings
    try:
        import youtube_transcript_api._settings as yts_settings
    except Exception:
        yts_settings = None  # if settings module isn't available, fall back to single attempt

    TranscriptsDisabled = getattr(yta, "TranscriptsDisabled", Exception)
    NoTranscriptFound = getattr(yta, "NoTranscriptFound", Exception)
    CouldNotRetrieveTranscript = getattr(yta, "CouldNotRetrieveTranscript", Exception)

    def _looks_like_dns_error(e: Exception) -> bool:
        s = (str(e) or "").lower()
        r = (repr(e) or "").lower()
        needles = [
            "failed to resolve",
            "nameresolutionerror",
            "no address associated with hostname",
            "temporary failure in name resolution",
            "name or service not known",
            "getaddrinfo failed",
            "gaierror",
            "errno -5",
        ]
        return any(n in s for n in needles) or any(n in r for n in needles)

    def _set_host(host: str) -> None:
        if not yts_settings:
            return
        # Known constants in youtube_transcript_api/_settings.py:
        # WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
        # INNERTUBE_API_URL = "https://www.youtube.com/youtubei/v1/player?key={api_key}"
        if hasattr(yts_settings, "WATCH_URL"):
            yts_settings.WATCH_URL = f"https://{host}/watch?v={{video_id}}"
        if hasattr(yts_settings, "INNERTUBE_API_URL"):
            yts_settings.INNERTUBE_API_URL = f"https://{host}/youtubei/v1/player?key={{api_key}}"

    # Put the most likely-to-work hosts first
    hosts = ["youtube.com", "m.youtube.com", "www.youtube.com"] if yts_settings else ["www.youtube.com"]

    last_err: Exception | None = None
    fetched = None

    for host in hosts:
        _set_host(host)
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(vid, languages=preferred_langs)  # ✅ new API
            break  # success
        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript found for this video.")
        except CouldNotRetrieveTranscript as e:
            # If it's a DNS/resolve failure, try the next host
            if _looks_like_dns_error(e):
                last_err = e
                continue
            raise ValueError(f"Could not retrieve transcript (blocked / unavailable): {e}")
        except Exception as e:
            # If it's a DNS/resolve failure, try the next host
            if _looks_like_dns_error(e):
                last_err = e
                continue
            raise RuntimeError(f"Unexpected YouTube transcript error: {e}")

    if fetched is None:
        # All hosts failed (most commonly: HF Spaces can't resolve/reach YouTube)
        if last_err is not None:
            raise RuntimeError(
                "YouTube transcript fetch failed because this environment can't resolve/reach YouTube "
                f"(tried: {', '.join(hosts)}). Last error: {last_err}"
            )
        raise RuntimeError(
            "YouTube transcript fetch failed (no transcript fetched) after trying: "
            + ", ".join(hosts)
        )

    # fetched is a FetchedTranscript object (iterable of snippet objects)
    try:
        return "\n".join(snippet.text for snippet in fetched if getattr(snippet, "text", None))
    except Exception:
        # fallback: raw list of dicts if the library returns that in some edge case
        if hasattr(fetched, "to_raw_data"):
            raw = fetched.to_raw_data()
            return "\n".join(x.get("text", "") for x in raw if x.get("text"))
        raise


def _fetch_webpage_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _chunk_text(text: str, chunk_chars: int = 5000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _map_summarize(llm, chunk: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Summarize this chunk concisely.
- 4-6 bullet points
- capture key facts, names, numbers if present

CHUNK:
{chunk}
"""
    )
    return (prompt | llm | StrOutputParser()).invoke({"chunk": chunk})


def _reduce_summaries(llm, summaries: List[str]) -> str:
    joined = "\n\n".join(summaries)
    prompt = ChatPromptTemplate.from_template(
        """Create a final summary from the chunk summaries:
- 3-5 sentence overview
- 6-10 bullet key points
- Actionable takeaways (3 bullets)

CHUNK SUMMARIES:
{summaries}
"""
    )
    return (prompt | llm | StrOutputParser()).invoke({"summaries": joined})


def summarize_url(llm, url: str) -> str:
    kind = detect_url_type(url)
    if kind == "invalid":
        raise ValueError("Please enter a valid URL.")

    if kind == "youtube":
        content = _fetch_youtube_transcript(url)
    else:
        content = _fetch_webpage_text(url)

    chunks = _chunk_text(content, chunk_chars=5000)
    if not chunks:
        raise ValueError("Could not extract readable text from the URL.")

    # Cap chunks for speed in demo apps
    chunk_summaries = [_map_summarize(llm, c) for c in chunks[:6]]
    return _reduce_summaries(llm, chunk_summaries)
