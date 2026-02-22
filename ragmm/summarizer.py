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
    import html
    import xml.etree.ElementTree as ET

    vid = _extract_youtube_id(url)
    if not vid:
        raise ValueError("Could not extract YouTube video ID from the URL.")

    preferred_langs = preferred_langs or ["en", "en-US", "en-GB"]

    # Build a preference order like: ["en-us","en-gb","en","en"] (deduped)
    def _lang_prefs(langs: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for l in langs:
            if not l:
                continue
            ll = l.strip().lower()
            if ll and ll not in seen:
                out.append(ll); seen.add(ll)
        # also add base languages
        for l in list(out):
            base = l.split("-")[0]
            if base and base not in seen:
                out.append(base); seen.add(base)
        return out

    prefs = _lang_prefs(preferred_langs)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

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

    def _parse_timedtext_xml(xml_text: str) -> str:
        xml_text = (xml_text or "").strip()
        if not xml_text:
            return ""
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return ""

        parts: List[str] = []
        # Typical format: <transcript><text start="..." dur="...">Hello</text>...</transcript>
        for node in root.findall(".//text"):
            t = node.text or ""
            if not t:
                continue
            t = html.unescape(t)
            t = t.replace("\n", " ").replace("\r", " ").strip()
            if t:
                parts.append(t)

        return "\n".join(parts).strip()

    def _timedtext_get(params: dict) -> str:
        # Uses "video.google.com" to avoid youtube.com DNS issues
        r = requests.get(
            "https://video.google.com/timedtext",
            params=params,
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        return r.text or ""

    def _timedtext_list_tracks() -> List[dict]:
        try:
            xml_text = _timedtext_get({"type": "list", "v": vid})
        except Exception:
            return []

        xml_text = (xml_text or "").strip()
        if not xml_text:
            return []

        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return []

        tracks: List[dict] = []
        for t in root.findall(".//track"):
            lang = (t.attrib.get("lang_code") or t.attrib.get("lang") or "").strip()
            name = (t.attrib.get("name") or "").strip()
            kind = (t.attrib.get("kind") or "").strip()  # often "asr" for auto captions
            if lang:
                tracks.append({"lang": lang, "name": name, "kind": kind})
        return tracks

    def _score_track(track: dict) -> tuple:
        # Lower is better
        lang = (track.get("lang") or "").lower()
        kind = (track.get("kind") or "").lower()
        # prefer non-asr (manual) over asr
        kind_penalty = 1 if kind == "asr" else 0
        # prefer languages earlier in prefs
        try:
            lang_rank = prefs.index(lang)
        except ValueError:
            # allow base matching (e.g. track "en" vs pref "en-us")
            base = lang.split("-")[0]
            lang_rank = prefs.index(base) if base in prefs else 10_000
        # prefer unnamed tracks slightly (often "standard" track)
        name_penalty = 1 if (track.get("name") or "").strip() else 0
        return (lang_rank, kind_penalty, name_penalty)

    # ---------------------------------------------------------------------
    # 1) Primary: Google timedtext (manual first, then auto/asr)
    # ---------------------------------------------------------------------
    timedtext_error: Exception | None = None
    try:
        tracks = _timedtext_list_tracks()

        # If we got tracks, try them in best-first order
        if tracks:
            tracks_sorted = sorted(tracks, key=_score_track)
            for tr in tracks_sorted:
                params = {"v": vid, "lang": tr["lang"]}
                if tr.get("name"):
                    params["name"] = tr["name"]
                if tr.get("kind"):
                    params["kind"] = tr["kind"]

                try:
                    xml_text = _timedtext_get(params)
                    text = _parse_timedtext_xml(xml_text)
                    if text:
                        return text
                except Exception as e:
                    # ignore and try next track
                    timedtext_error = e

        # If list endpoint returned nothing (or none worked), try direct preferred langs
        # First: manual captions
        for lang in prefs:
            try:
                xml_text = _timedtext_get({"v": vid, "lang": lang})
                text = _parse_timedtext_xml(xml_text)
                if text:
                    return text
            except Exception as e:
                timedtext_error = e

        # Then: auto captions (ASR)
        for lang in prefs:
            try:
                xml_text = _timedtext_get({"v": vid, "lang": lang, "kind": "asr"})
                text = _parse_timedtext_xml(xml_text)
                if text:
                    return text
            except Exception as e:
                timedtext_error = e

    except Exception as e:
        timedtext_error = e

    # ---------------------------------------------------------------------
    # 2) Fallback: youtube_transcript_api (your original path)
    # ---------------------------------------------------------------------
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(vid, languages=preferred_langs)
    except getattr(yta, "TranscriptsDisabled", Exception):
        raise ValueError("Transcripts are disabled for this video.")
    except getattr(yta, "NoTranscriptFound", Exception):
        raise ValueError("No transcript found for this video.")
    except getattr(yta, "CouldNotRetrieveTranscript", Exception) as e:
        # If timedtext failed due to DNS/egress too, surface that context
        if timedtext_error and _looks_like_dns_error(timedtext_error):
            raise ValueError(
                "Could not retrieve transcript. This hosting environment appears to have DNS/egress "
                "restrictions to YouTube/Google caption endpoints."
            ) from e
        raise ValueError(f"Could not retrieve transcript (blocked / unavailable): {e}")
    except Exception as e:
        # If timedtext failed too, give a more helpful combined error
        if timedtext_error:
            raise RuntimeError(
                f"Transcript fetch failed via timedtext and youtube_transcript_api. "
                f"timedtext last error: {timedtext_error} | yta error: {e}"
            )
        raise RuntimeError(f"Unexpected YouTube transcript error: {e}")

    try:
        return "\n".join(snippet.text for snippet in fetched if getattr(snippet, "text", None))
    except Exception:
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
