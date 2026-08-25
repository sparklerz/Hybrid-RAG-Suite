from __future__ import annotations
from typing import Literal, List
import os
import re

import validators
import requests
from bs4 import BeautifulSoup

try:  # optional: only used as a last-resort fallback
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:  # pragma: no cover - package missing or broken
    YouTubeTranscriptApi = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def detect_url_type(url: str) -> Literal["youtube", "web", "invalid"]:
    if not url or not validators.url(url):
        return "invalid"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "web"


def _extract_youtube_id(url: str) -> str | None:
    # supports watch?v=, youtu.be/, /shorts/, /embed/, /live/, /v/
    patterns = [
        r"[?&]v=([A-Za-z0-9_\-]{11})",
        r"youtu\.be/([A-Za-z0-9_\-]{11})",
        r"/(?:shorts|embed|live|v)/([A-Za-z0-9_\-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    m = re.search(r"(?:v=|youtu\.be/|/)([A-Za-z0-9_\-]{11})(?:[?&/#]|$)", url)
    return m.group(1) if m else None


# YouTube drops TLS connections on us fairly often (SSLEOFError) and rate-limits
# datacenter IPs, so every outbound call goes through a retrying session.
_YT_SESSION = None


def _yt_session():
    global _YT_SESSION
    if _YT_SESSION is not None:
        return _YT_SESSION

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    proxy = os.environ.get("YT_PROXY") or os.environ.get("YOUTUBE_PROXY")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})

    _YT_SESSION = s
    return s


def _is_network_error(e: BaseException) -> bool:
    import ssl as _ssl

    if isinstance(e, (requests.exceptions.SSLError,
                      requests.exceptions.ConnectionError,
                      requests.exceptions.Timeout,
                      _ssl.SSLError,
                      OSError)):
        return True
    s = f"{e!r} {e}".lower()
    return any(n in s for n in ("unexpected_eof", "ssleoferror", "max retries exceeded",
                                "connection reset", "connection aborted", "handshake"))


def _retrying(fn, attempts: int = 3, base_delay: float = 1.5):
    """Run fn(), retrying transient TLS/connection failures with backoff."""
    import time

    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if not _is_network_error(e) or i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


_UA_DESKTOP = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
_UA_ANDROID = "com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip"

# Public InnerTube key. The WEB client is answered with UNPLAYABLE for caption
# requests, so we ask as the mobile apps, which still serve caption tracks.
_INNERTUBE_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
_INNERTUBE_CLIENTS = (
    ({"clientName": "ANDROID", "clientVersion": "20.10.38",
      "androidSdkVersion": 30, "hl": "en"}, _UA_ANDROID, "3"),
    ({"clientName": "IOS", "clientVersion": "20.10.4",
      "deviceModel": "iPhone16,2", "hl": "en"}, _UA_DESKTOP, "5"),
)


def _fetch_youtube_transcript(url: str, preferred_langs: List[str] | None = None) -> str:
    import html
    import json
    import xml.etree.ElementTree as ET

    vid = _extract_youtube_id(url)
    if not vid:
        raise ValueError("Could not extract YouTube video ID from the URL.")

    preferred_langs = preferred_langs or ["en", "en-US", "en-GB"]

    def _lang_prefs(langs: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for l in langs:
            if not l:
                continue
            ll = l.strip().lower()
            if ll and ll not in seen:
                out.append(ll)
                seen.add(ll)
        for l in list(out):
            base = l.split("-")[0]
            if base and base not in seen:
                out.append(base)
                seen.add(base)
        return out

    prefs = _lang_prefs(preferred_langs)
    sess = _yt_session()
    net_errors: List[Exception] = []
    block_reason: str | None = None

    # ------------------------------------------------------------
    # transcript parsing: json3, srv3 (<p>/<s>) and legacy <transcript>
    # ------------------------------------------------------------
    def _parse_json3(text: str) -> str:
        try:
            data = json.loads(text)
        except Exception:
            return ""
        lines: List[str] = []
        for ev in data.get("events") or []:
            seg = "".join((s.get("utf8") or "") for s in (ev.get("segs") or []))
            seg = seg.replace("\n", " ").strip()
            if seg:
                lines.append(seg)
        return "\n".join(lines).strip()

    def _parse_timedtext(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if text.startswith("{"):
            return _parse_json3(text)
        try:
            root = ET.fromstring(text)
        except Exception:
            return ""

        parts: List[str] = []
        # srv3: <body><p><s>word</s>...</p></body>
        for p in root.findall(".//body/p"):
            t = html.unescape("".join(p.itertext()))
            t = t.replace("\n", " ").replace("\r", " ").strip()
            if t:
                parts.append(t)
        if parts:
            return "\n".join(parts).strip()

        # legacy: <transcript><text start=...>...</text></transcript>
        for node in root.findall(".//text"):
            t = node.text or ""
            if not t:
                continue
            t = html.unescape(t).replace("\n", " ").replace("\r", " ").strip()
            if t:
                parts.append(t)
        return "\n".join(parts).strip()

    # ------------------------------------------------------------
    # caption track discovery
    # ------------------------------------------------------------
    def _tracks_from_player(player: dict) -> List[dict]:
        renderer = (player.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
        tracks: List[dict] = []
        for t in renderer.get("captionTracks") or []:
            base = t.get("baseUrl")
            if not base:
                continue
            tracks.append({
                "baseUrl": base,
                "lang": (t.get("languageCode") or "").strip(),
                "kind": (t.get("kind") or "").strip(),   # "asr" for auto-generated
                "name": ((t.get("name") or {}).get("simpleText") or "").strip(),
            })
        return tracks

    def _innertube_tracks() -> List[dict]:
        nonlocal block_reason
        for client, ua, client_id in _INNERTUBE_CLIENTS:
            try:
                def _call(client=client, ua=ua, client_id=client_id):
                    r = sess.post(
                        "https://www.youtube.com/youtubei/v1/player?key=" + _INNERTUBE_KEY,
                        json={"context": {"client": client}, "videoId": vid,
                              "contentCheckOk": True, "racyCheckOk": True},
                        headers={"User-Agent": ua,
                                 "Content-Type": "application/json",
                                 "X-YouTube-Client-Name": client_id,
                                 "X-YouTube-Client-Version": client["clientVersion"],
                                 "Accept-Language": "en-US,en;q=0.9"},
                        timeout=25,
                    )
                    r.raise_for_status()
                    return r.json()

                player = _retrying(_call)
            except Exception as e:
                if _is_network_error(e):
                    net_errors.append(e)
                continue

            status = player.get("playabilityStatus") or {}
            if status.get("status") not in (None, "OK"):
                block_reason = status.get("reason") or status.get("status")

            tracks = _tracks_from_player(player)
            if tracks:
                return tracks
        return []

    def _watchpage_tracks() -> List[dict]:
        try:
            def _call():
                r = sess.get(
                    "https://www.youtube.com/watch?v=" + vid,
                    headers={"User-Agent": _UA_DESKTOP,
                             "Accept-Language": "en-US,en;q=0.9"},
                    cookies={"CONSENT": "YES+cb", "SOCS": "CAI"},
                    timeout=25,
                )
                r.raise_for_status()
                return r.text

            page = _retrying(_call)
        except Exception as e:
            if _is_network_error(e):
                net_errors.append(e)
            return []

        m = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;\s*(?:var\s|</script>)", page, re.S)
        if not m:
            return []
        try:
            return _tracks_from_player(json.loads(m.group(1)))
        except Exception:
            return []

    def _score_track(track: dict) -> tuple:
        lang = (track.get("lang") or "").lower()
        kind_penalty = 1 if (track.get("kind") or "").lower() == "asr" else 0
        try:
            lang_rank = prefs.index(lang)
        except ValueError:
            base = lang.split("-")[0]
            lang_rank = prefs.index(base) if base in prefs else 10_000
        return (lang_rank, kind_penalty)

    def _download(track: dict) -> str:
        for suffix in ("&fmt=json3", ""):
            try:
                def _call(u=track["baseUrl"] + suffix):
                    r = sess.get(u, headers={"User-Agent": _UA_DESKTOP,
                                             "Accept-Language": "en-US,en;q=0.9"},
                                 timeout=25)
                    r.raise_for_status()
                    return r.text or ""

                text = _parse_timedtext(_retrying(_call))
                if text:
                    return text
            except Exception as e:
                if _is_network_error(e):
                    net_errors.append(e)
        return ""

    for discover in (_innertube_tracks, _watchpage_tracks):
        tracks = discover()
        if not tracks:
            continue
        for tr in sorted(tracks, key=_score_track):
            text = _download(tr)
            if text:
                return text

    # ------------------------------------------------------------
    # last resort: youtube_transcript_api (scrapes youtube.com directly)
    # ------------------------------------------------------------
    if YouTubeTranscriptApi is not None:
        try:
            fetched = _retrying(
                lambda: YouTubeTranscriptApi().fetch(vid, languages=preferred_langs)
            )
            text = "\n".join(s.text for s in fetched if getattr(s, "text", None)).strip()
            if text:
                return text
        except Exception as e:
            name = type(e).__name__
            if name == "TranscriptsDisabled":
                raise ValueError("Transcripts are disabled for this video.")
            if name == "NoTranscriptFound":
                raise ValueError(
                    "No transcript found for this video in "
                    + ", ".join(preferred_langs) + "."
                )
            if _is_network_error(e):
                net_errors.append(e)
            elif not block_reason:
                block_reason = str(e)

    if net_errors:
        last = net_errors[-1]
        raise RuntimeError(
            "Could not reach YouTube to download captions - the connection kept failing "
            f"({type(last).__name__}: {last}). This is usually YouTube rejecting requests "
            "from this IP address. Retry in a moment, or set the YT_PROXY environment "
            "variable to route caption requests through a proxy."
        )

    if block_reason:
        raise ValueError(f"YouTube refused to serve captions for this video: {block_reason}")

    raise ValueError(
        "No captions are available for this video (looked for "
        + ", ".join(preferred_langs) + ")."
    )


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
