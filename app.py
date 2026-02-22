from __future__ import annotations

import os
import time
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory

from ragmm.settings import Settings
from ragmm.factories import get_llm, get_embeddings
from ragmm.ingest import load_uploaded_pdfs, load_pdf_folder, split_docs
from ragmm.vectorstore import (
    get_chroma,
    list_collections,
    rebuild_collection,
    delete_collection,
    sanitize_collection_name
)
from ragmm.rag import make_conversational_rag
from ragmm.web_search import search_web_wiki_arxiv, answer_with_sources
from ragmm.summarizer import summarize_url
from ragmm.chunk_ids import ensure_chunk_ids
from ragmm.artifacts import save_docstore, delete_bm25_cache, load_docstore
from ragmm.bm25_retriever import BM25Retriever
from ragmm.hybrid_retriever import HybridRetriever
from ragmm.collections import collection_stats, available_collections, delete_collection_artifacts
from ragmm.ui import section
from ragmm.rerank import CrossEncoderReranker, RerankRetriever
from pathlib import Path
import hashlib

st.set_page_config(page_title="Hybrid RAG Suite", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stChatMessageContainer"] { max-width: 900px; margin: 0 auto; }
    div[data-testid="stChatInput"] { max-width: 900px; margin: 0 auto; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def session_key(page_name: str, collection: str) -> str:
    return f"{page_name}::{collection}"

def get_history_store() -> dict[str, dict[str, ChatMessageHistory]]:
    if "history_store" not in st.session_state:
        st.session_state.history_store = {"kb": {}, "uploads": {}, "other": {}}
    return st.session_state.history_store

def ensure_chat_registry():
    if "chat_registry" not in st.session_state:
        st.session_state.chat_registry = {}


def get_chat_sessions_for_collection(collection: str) -> list[str]:
    ensure_chat_registry()
    return st.session_state.chat_registry.get(collection, [])


def add_chat_session_for_collection(collection: str, session_id: str):
    ensure_chat_registry()
    st.session_state.chat_registry.setdefault(collection, [])
    if session_id not in st.session_state.chat_registry[collection]:
        st.session_state.chat_registry[collection].insert(0, session_id)

def get_or_create_history(session_id: str) -> ChatMessageHistory:
    stores = get_history_store()
    if session_id.startswith("kb::"):
        store = stores["kb"]
    elif session_id.startswith("uploads::"):
        store = stores["uploads"]
    else:
        store = stores["other"]

    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def sources_panel(context_docs):
    with st.expander("Sources (retrieved chunks)", expanded=False):
        for i, d in enumerate(context_docs or [], start=1):
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("file_name") or meta.get("source_type") or "unknown"
            page = meta.get("page", meta.get("page_number", ""))

            retr = meta.get("retriever", "dense")
            bm25_s = meta.get("bm25_score", None)
            hyb_s = meta.get("hybrid_score", None)
            rerank_s = meta.get("rerank_score", None)

            score_txt = ""
            if bm25_s is not None:
                score_txt += f" | bm25={bm25_s:.3f}"
            if hyb_s is not None:
                score_txt += f" | hybrid={hyb_s:.6f}"
            if rerank_s is not None:
                score_txt += f" | rerank={rerank_s:.4f}"

            # st.markdown(f"**#{i}** — `{src}` {('page ' + str(page)) if page != '' else ''}  \n"
            cid = (meta.get("chunk_id") or "").strip()
            cid_txt = f" | chunk_id=`{cid}`" if cid else ""
            st.markdown(
                f"**#{i}** — `{src}` {('page ' + str(page)) if page != '' else ''}  \n"
                f"**retriever:** `{retr}`{score_txt}{cid_txt}"
            )
            st.write(d.page_content)
            st.divider()


settings = Settings()

# ---------------- Startup cleanup: keep only kb_default across restarts -----------------
PERSISTENT_COLLECTIONS = {"kb_default"}  # only this survives app restarts

def cleanup_temporary_collections_once_per_process(settings):
    """
    Delete all non-persistent collections exactly once per Streamlit server process.
    (So it doesn't wipe again on every rerun or new browser tab.)
    """
    marker_dir = Path(getattr(settings, "artifacts_dir", settings.chroma_dir))
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / ".startup_cleanup_pid"

    pid = str(os.getpid())
    prev = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    if prev == pid:
        return

    marker.write_text(pid, encoding="utf-8")

    for name, _count in list_collections(settings):
        if name not in PERSISTENT_COLLECTIONS:
            try:
                delete_collection_artifacts(settings, name)
            except Exception:
                pass
            try:
                delete_collection(settings, name)
            except Exception:
                pass

cleanup_temporary_collections_once_per_process(settings)

# ---------------- Sidebar ----------------
st.sidebar.title("📚 Hybrid RAG Suite")

page = st.sidebar.radio(
    "Navigation",
    [
        "Document Chat (Uploads)",
        "Knowledge Base",
        "Web Search Chat",
        "Summarizer",
        "RAG Evaluation",
        "Collection Manager",
    ],
    index=0,
)

# Track page switches (used to prime KB so old page UI doesn't linger during heavy indexing)
prev_page = st.session_state.get("_last_page")
if prev_page != page:
    st.session_state["_last_page"] = page
    st.session_state["_page_switched"] = True

st.sidebar.subheader("Global Settings")
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

default_model = settings.llm_model
default_index = GROQ_MODELS.index(default_model) if default_model in GROQ_MODELS else 0

llm_model = st.sidebar.selectbox(
    "LLM model (Groq)",
    GROQ_MODELS,
    index=default_index,
)

embed_provider = st.sidebar.selectbox("Embeddings", ["huggingface"], index=0)
retriever_mode = st.sidebar.selectbox(
    "Retriever Mode",
    ["dense", "bm25", "hybrid"],
    index=2,
)
top_k = st.sidebar.slider("Top-K chunks", min_value=3, max_value=12, value=settings.top_k, step=1)
st.sidebar.subheader("Reranker (Optional)")
use_reranker = st.sidebar.checkbox("Enable cross-encoder reranker", value=False)
rerank_model = st.sidebar.text_input(
    "Reranker model",
    value="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
fetch_k = st.sidebar.slider("Fetch-K before rerank", 8, 30, 12, 1)
rerank_top_k = st.sidebar.slider("Top-K after rerank", 3, 12, top_k, 1)

# apply sidebar overrides
settings = Settings(
    chroma_dir=settings.chroma_dir,
    groq_api_key=settings.groq_api_key,
    llm_model=llm_model,
    embed_provider=embed_provider,
    hf_token=settings.hf_token,
    kb_pdf_dir=settings.kb_pdf_dir,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    top_k=top_k,
)

st.sidebar.caption(f"Chroma DB: {settings.chroma_dir}")

@st.cache_resource
def _reranker_cached(model_name: str):
    return CrossEncoderReranker(model_name)

# cached singletons
@st.cache_resource
def _embeddings_cached(embed_provider: str):
    # embed_provider affects embedding object
    return get_embeddings(settings)

# Note: LLM is lightweight; don’t cache across users by default
def make_retriever(collection_name: str):
    embeddings = _embeddings_cached(settings.embed_provider)
    vs = get_chroma(settings, embeddings, collection_name)

    dense = vs.as_retriever(search_kwargs={"k": settings.top_k})

    if retriever_mode == "dense":
        base = dense
    elif retriever_mode == "bm25":
        bm25 = BM25Retriever(settings, collection_name, k=settings.top_k)
        base = bm25
    else:
        bm25 = BM25Retriever(settings, collection_name, k=settings.top_k)
        base = HybridRetriever(
                    dense_retriever=dense,
                    bm25_retriever=bm25,
                    k=settings.top_k,
                    dense_k=max(settings.top_k, 8),
                    bm25_k=max(settings.top_k, 8),
                    w_dense=1.0,
                    w_bm25=1.0,
                )

    if use_reranker:
        rr = _reranker_cached(rerank_model)
        return RerankRetriever(base, rr, fetch_k=fetch_k, top_k=rerank_top_k)

    return base

def _kb_chat_label(sid: str) -> str:
    parts = (sid or "").split("::")
    suffix = parts[-1] if parts else (sid[-6:] if sid else "chat")
    stamp = parts[-2] if len(parts) >= 2 else ""
    # stamp: YYYYMMDD_HHMMSS -> YYYY-MM-DD HH:MM
    if "_" in stamp and len(stamp) >= 15:
        d, t = stamp.split("_", 1)
        stamp = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}"
    return f"{stamp} • {suffix}" if stamp else f"Chat • {suffix}"

def kb_signature(folder: str) -> str:
    p = Path(folder)
    rows = []
    for f in sorted(p.glob("*.pdf")):
        try:
            stt = f.stat()
            rows.append(f"{f.name}|{stt.st_size}|{int(stt.st_mtime)}")
        except Exception:
            rows.append(f"{f.name}|err")
    blob = "\n".join(rows).encode("utf-8", errors="ignore")
    return hashlib.md5(blob).hexdigest()

def _prompt_sig(session_id: str, prompt: str) -> str:
    h = hashlib.md5(prompt.strip().encode("utf-8")).hexdigest()
    return f"{session_id}:{h}"

# ---------------- Page: Upload Document Chat ----------------
if page == "Document Chat (Uploads)":
    st.title("📄 Document Chat (Uploads)")
    st.write(
        "Upload PDFs to create a **new document set** and chat with it. "
        "Each document set has **one chat** tied to it."
    )

    # --- list upload docsets (everything except persistent) ---
    all_cols = available_collections(settings)
    upload_docsets = [c for c in all_cols if c not in PERSISTENT_COLLECTIONS]
    CREATING_PLACEHOLDER = "🆕 New document set (not indexed yet)"

    # --- Apply pending docset selection BEFORE selectbox is created ---
    pending_choice = st.session_state.pop("uploads_active_docset_pending_choice", None)
    if pending_choice is not None:
        st.session_state["uploads_active_docset"] = pending_choice

    # one-time init
    if "uploads_locked" not in st.session_state:
        st.session_state["uploads_locked"] = False

    # Pending selection (used after indexing a new docset)
    pending = st.session_state.pop("uploads_active_docset_pending", None)
    if pending and pending in upload_docsets:
        st.session_state["uploads_active_docset"] = pending

    # default selection (only if not in "creating new docset" mode)
    if upload_docsets and "uploads_active_docset" not in st.session_state and not st.session_state.get("uploads_creating_new", False):
        st.session_state["uploads_active_docset"] = upload_docsets[-1]

    uploads_nonce = st.session_state.get("uploads_nonce", 0)

    # planned new docset id (stable until you click Start New again)
    if "uploads_planned_docset" not in st.session_state:
        st.session_state["uploads_planned_docset"] = sanitize_collection_name(f"uploads_{time.strftime('%Y%m%d_%H%M%S')}")
    planned_docset = st.session_state["uploads_planned_docset"]

    left, right = st.columns([2, 1], gap="large")

    # ---------- Right: select docset + show summary ----------
    with right:
        section("Document set")

        if upload_docsets:
            creating_new = st.session_state.get("uploads_creating_new", False)
            docset_options = ([CREATING_PLACEHOLDER] + upload_docsets) if creating_new else upload_docsets

            active_docset = st.selectbox(
                "Choose a document set",
                options=docset_options,
                key="uploads_active_docset",
                help="Selecting a document set loads its chat (1 docset ⇄ 1 chat).",
                disabled=creating_new,   # optional, but matches your intention of “wipe while creating”
            )

            # ---- Docset summary from docstore ----
            docs = None
            try:
                if active_docset == CREATING_PLACEHOLDER:
                    st.caption("Creating a new document set… it will appear here after you index PDFs.")
                else:
                    docs = load_docstore(settings, active_docset)
            except Exception:
                docs = None

            if docs:
                from collections import defaultdict

                by_file = defaultdict(lambda: {"chunks": 0, "pages": set()})
                for d in docs:
                    meta = getattr(d, "metadata", {}) or {}
                    fname = meta.get("file_name") or meta.get("source") or "unknown"
                    by_file[fname]["chunks"] += 1
                    page = meta.get("page", meta.get("page_number", None))
                    if page is not None:
                        by_file[fname]["pages"].add(page)

                st.caption(f"📄 **{len(by_file)} PDF(s)** · 🧩 **{len(docs)} chunks**")

                with st.expander("Files in this document set", expanded=False):
                    for fname, s in sorted(by_file.items(), key=lambda x: x[0].lower()):
                        pages = len(s["pages"]) if s["pages"] else "?"
                        st.write(f"- `{fname}` — {s['chunks']} chunks, pages: {pages}")
            else:
                st.caption("No file summary available yet (docstore not found for this set).")

        else:
            st.info("No upload document sets yet. Create one on the left.")

        st.caption("Tip: use **Collection Manager** if you want to delete older document sets.")

    # ---------- Left: create NEW docset ----------
    with left:
        st.markdown("#### Create a new document set")

        st.caption(f"New document set ID: `{planned_docset}`")

        files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploads_files::{uploads_nonce}",
            disabled=st.session_state["uploads_locked"],
        )

        index_status = st.empty()
        b1, b2 = st.columns(2)
        with b1:
            can_start_new = bool(upload_docsets)
            if st.button("➕ Start new document set", use_container_width=True, disabled=not can_start_new):
                st.session_state["uploads_creating_new"] = True

                # ✅ unlock uploader again
                st.session_state["uploads_locked"] = False

                # ✅ set dropdown to a "blank/new" placeholder
                st.session_state["uploads_active_docset_pending_choice"] = CREATING_PLACEHOLDER

                # New planned id + clear uploader
                st.session_state["uploads_planned_docset"] = sanitize_collection_name(
                    f"uploads_{time.strftime('%Y%m%d_%H%M%S')}"
                )
                st.session_state["uploads_nonce"] = uploads_nonce + 1
                st.rerun()

        with b2:
            if st.button(
                "Index PDFs",
                type="primary",
                use_container_width=True,
                disabled=st.session_state["uploads_locked"] or (not files),
                key=f"uploads_index::{uploads_nonce}",
            ):
                try:
                    with index_status.container():
                        with st.spinner("Indexing PDFs..."):
                            embeddings = _embeddings_cached(settings.embed_provider)
                            raw_docs = load_uploaded_pdfs(files)
                            chunks = ensure_chunk_ids(split_docs(raw_docs, settings))

                            n = rebuild_collection(settings, embeddings, planned_docset, chunks)
                            save_docstore(settings, planned_docset, chunks)

                        # lock uploads AFTER indexing so uploader gets disabled
                        st.session_state["uploads_locked"] = True

                        # select new docset next render + fresh chat
                        st.session_state["uploads_active_docset_pending"] = planned_docset
                        # get_history_store().pop(planned_docset, None)
                        get_history_store().pop(f"uploads::{planned_docset}", None)

                        # allow chatting now
                        st.session_state["uploads_creating_new"] = False

                        st.success(f"Indexed {n} chunks into `{planned_docset}`.")
                        st.rerun()

                except Exception as e:
                    st.error(str(e))

        if st.session_state.get("uploads_locked", False):
                st.info("Document set indexed. Click **Start new document set** to upload another set of PDFs.")

    # CHAT OUTSIDE columns (fixes the “not aligned / awkward bottom input” feel)
    st.divider()
    section("Chat")

    if st.session_state.get("uploads_creating_new", False):
        st.info("You’re creating a new document set. Upload PDFs and click **Index PDFs** to start chatting.")
        st.stop()

    active_docset = st.session_state.get("uploads_active_docset")
    if not active_docset:
        st.info("Select a document set on the right, or index PDFs to create one.")
        st.stop()

    uploads_sid = f"uploads::{active_docset}"
    hist = get_or_create_history(uploads_sid)

    if not getattr(hist, "messages", []):
        st.chat_message("assistant").write("Ask a question about your uploaded PDFs.")

    for msg in getattr(hist, "messages", []):
        msg_type = getattr(msg, "type", "") or msg.__class__.__name__.lower()
        role = "user" if "human" in msg_type else "assistant"
        st.chat_message(role).write(getattr(msg, "content", str(msg)))

    prompt = st.chat_input(
        "Ask a question about your PDFs...",
        key=f"uploads_chat_input::{active_docset}",
    )

    if prompt:
        # --- Chat-meta shortcut (so "what is my first question?" works) ---
        p = prompt.strip().lower()
        meta_q = any(x in p for x in ["my first question", "what did i ask first", "first thing i asked"])
        if meta_q:
            first_user = None
            for m in getattr(hist, "messages", []):
                t = getattr(m, "type", "") or m.__class__.__name__.lower()
                if "human" in t:
                    first_user = getattr(m, "content", "")
                    break

            st.chat_message("user").write(prompt)
            answer = first_user or "I don't see any earlier user questions in this chat yet."
            hist.add_user_message(prompt)
            hist.add_ai_message(answer)
            st.chat_message("assistant").write(answer)
            st.stop()

        st.chat_message("user").write(prompt)

        try:
            llm = get_llm(settings, model=llm_model, streaming=False)
            retriever = make_retriever(active_docset)
            convo = make_conversational_rag(llm, retriever, get_or_create_history)

            with st.chat_message("assistant"):
                answer_slot = st.empty()
                with st.spinner("Thinking..."):
                    out = convo.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": uploads_sid}},
                    )

                answer_slot.write(out.get("answer", ""))
                sources_panel(out.get("context", []))

        except Exception as e:
            st.error(str(e))

# ---------------- Page: Knowledge Base ----------------
elif page == "Knowledge Base":
    st.title("🏛️ Knowledge Base — Thirukkural")

    # PRIME RUN: finish a fast render first so Streamlit clears previous page UI,
    # then immediately rerun (the 2nd run can do the heavier indexing safely).
    if st.session_state.pop("_page_switched", False):
        st.caption("Loading Knowledge Base…")
        st.rerun()
        st.stop()

    st.write(
        "This KB (`kb_default`) contains **Thirukkural** (English translation + meaning PDF). "
        "Tamil couplets are fetched from the Thirukkural API by couplet number. "
        "Ask about a topic (eg: leadership, patience, charity) and I'll return **Tamil couplet(s) first**, then **English**."
    )

    kb_collection = "kb_default"

    # --- KB folder + show what’s inside ---
    kb_folder = st.text_input("KB PDF folder", value=settings.kb_pdf_dir, key="kb_folder", disabled=True)
    st.caption("To change this path, set KB_PDF_DIR in your .env / environment.")

    sig = kb_signature(kb_folder)
    last_sig = st.session_state.get("kb_last_sig")

    # Optional toggle (place BEFORE any widgets that use this key)
    auto_index = st.session_state.get("kb_auto_index", True)

    if auto_index and sig != last_sig:
        with st.spinner("KB changed — rebuilding index..."):
            embeddings = _embeddings_cached(settings.embed_provider)
            raw_docs = load_pdf_folder(kb_folder)
            chunks = ensure_chunk_ids(split_docs(raw_docs, settings))

            n = rebuild_collection(settings, embeddings, kb_collection, chunks)

            # keep artifacts consistent for BM25/hybrid
            save_docstore(settings, kb_collection, chunks)
            delete_bm25_cache(settings, kb_collection)  # so BM25 rebuilds cleanly next query

        st.session_state["kb_last_sig"] = sig
        st.success(f"KB indexed: {n} chunks (auto-updated).")
        # Make sure KB chat renders cleanly after indexing
        st.rerun()
    else:
        # keep stored signature if not present
        st.session_state.setdefault("kb_last_sig", sig)

    st.toggle("Auto-index KB when PDFs change", key="kb_auto_index", value=True)

    pdfs = sorted(Path(kb_folder).glob("*.pdf"))
    with st.expander("📚 What’s in this KB?", expanded=True):
        if pdfs:
            for p in pdfs:
                st.write(f"- `{p.name}`")
        else:
            st.warning("No PDFs found in this folder yet.")

    ex1 = "What does Thirukkural say about **leadership**?"
    ex2 = "What does Thirukkural say about **friendship**."
    ex3 = "What does Thirukkural say about **welfare of a nation**?"
    ex4 = "What does Thirukkural say about **gratitude**?"
    ex5 = "What does Thirukkural say about **patience**?"
    ex6 = "What does Thirukkural say about **charity**?"

    def set_kb_prompt(p: str):
        st.session_state["kb_pending_prompt"] = p
        st.rerun()

    r1c1, r1c2, r1c3 = st.columns(3)
    if r1c1.button("👑 Leadership", use_container_width=True):
        set_kb_prompt(ex1)
    if r1c2.button("🫂 Friendship", use_container_width=True):
        set_kb_prompt(ex2)
    if r1c3.button("🏛️ Welfare of a Nation", use_container_width=True):
        set_kb_prompt(ex3)

    r2c1, r2c2, r2c3 = st.columns(3)
    if r2c1.button("🙏 Gratitude", use_container_width=True):
        set_kb_prompt(ex4)
    if r2c2.button("🧘 Patience", use_container_width=True):
        set_kb_prompt(ex5)
    if r2c3.button("🤝 Charity", use_container_width=True):
        set_kb_prompt(ex6)

    st.divider()

    st.subheader("💬 KB Chats")

    sessions = get_chat_sessions_for_collection(kb_collection)
    if not sessions:
        import uuid
        new_id = f"{session_key('kb_chat', kb_collection)}::{time.strftime('%Y%m%d_%H%M%S')}::{uuid.uuid4().hex[:6]}"
        add_chat_session_for_collection(kb_collection, new_id)
        sessions = get_chat_sessions_for_collection(kb_collection)
        st.session_state["kb_active_session"] = new_id

    st.session_state.setdefault("kb_active_session", sessions[0])

    pending = st.session_state.pop("kb_active_session_pending", None)
    if pending is not None:
        st.session_state["kb_active_session"] = pending

    csel, cbtn = st.columns([4, 1], gap="small")

    with csel:
        active_session = st.selectbox(
            "Select a chat",
            options=sessions,
            index=sessions.index(st.session_state["kb_active_session"]) if st.session_state["kb_active_session"] in sessions else 0,
            key="kb_active_session",
            format_func=_kb_chat_label,
            label_visibility="collapsed",
        )

    with cbtn:
        if st.button("➕ New chat", use_container_width=True):
            import uuid
            new_id = f"{session_key('kb_chat', kb_collection)}::{time.strftime('%Y%m%d_%H%M%S')}::{uuid.uuid4().hex[:6]}"
            add_chat_session_for_collection(kb_collection, new_id)
            st.session_state["kb_active_session_pending"] = new_id
            st.rerun()
    
    st.divider()

    st.subheader("Chat with Thirukkural KB")

    kb_sid = f"kb::{active_session}"
    hist = get_or_create_history(kb_sid)

    # Greeting
    if not getattr(hist, "messages", []):
        st.chat_message("assistant").write(
            "Ask me anything about Thirukkural. Try the example buttons above, or ask your own question."
        )

    # Render history
    for msg in getattr(hist, "messages", []):
        msg_type = getattr(msg, "type", "") or msg.__class__.__name__.lower()
        role = "user" if "human" in msg_type else "assistant"
        st.chat_message(role).write(getattr(msg, "content", str(msg)))

    prompt = st.chat_input(
        "Ask about Thirukkural (topics, kurals, meanings)...",
        key=f"kb_chat_input::{kb_sid}",
    )

    if not prompt and st.session_state.get("kb_pending_prompt"):
        prompt = st.session_state.pop("kb_pending_prompt", None)

    if prompt:
        sig = _prompt_sig(kb_sid, prompt)
        now = time.time()
        last_sig = st.session_state.get("kb_last_prompt_sig")
        last_ts = st.session_state.get("kb_last_prompt_ts", 0.0)
        if sig == last_sig and (now - last_ts) < 2.0:
            st.stop()

        st.session_state["kb_last_prompt_sig"] = sig
        st.session_state["kb_last_prompt_ts"] = now

        try:
            st.chat_message("user").write(prompt)
            llm = get_llm(settings, model=llm_model, streaming=False)
            retriever = make_retriever(kb_collection)
            convo = make_conversational_rag(
                llm, retriever, get_or_create_history, answer_style="thirukkural"
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    convo.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": kb_sid}},
                    )
            st.rerun()

        except Exception as e:
            st.error(str(e))

# ---------------- Page: Collection Manager ----------------
elif page == "Collection Manager":
    st.title("🗂️ Collection Manager")
    st.write("View and manage collections stored in Chroma (persistent on disk).")

    section("Collections Overview", "Docstore and BM25 artifacts must exist for BM25/Hybrid to work.")

    stats = collection_stats(settings)
    if stats:
        for name, count, has_ds, has_bm25 in stats:
            with st.container(border=True):
                st.write(f"### `{name}`")
                st.write(f"- chunks: **{count}**")
                st.write(f"- docstore: **{'✅' if has_ds else '❌'}**")
                st.write(f"- bm25 cache: **{'✅' if has_bm25 else '❌ (created on first BM25 query)'}**")
    else:
        st.info("No collections found.")


    cols = list_collections(settings)
    if not cols:
        st.info("No collections found yet. Index uploads or build the KB first.")
    else:
        for name, count in cols:
            with st.container(border=True):
                c1, c2, c3 = st.columns([2.5, 1, 1])
                c1.markdown(f"### `{name}`")
                c2.metric("Chunks", count if count >= 0 else "unknown")

                with c3:
                    if st.button("Delete", key=f"del_{name}"):
                        try:
                            delete_collection(settings, name)
                            st.success(f"Deleted `{name}`. Refresh the page.")
                        except Exception as e:
                            st.error(str(e))

    st.caption("Tip: keep KB collections small for faster demo startup. Use Collection Manager to delete old upload collections.")

elif page == "Web Search Chat":
    st.title("🌐 Web Search Chat (Perplexity-style)")
    st.write("Multi-turn chat grounded on WebSearch + Wikipedia + arXiv snippets.")

    # ---- Multi-chat sessions (like KB) ----
    web_collection = "web_search"
    st.subheader("💬 Web Search Chats")

    sessions = get_chat_sessions_for_collection(web_collection)
    if not sessions:
        import uuid
        new_id = f"{session_key('web_chat', web_collection)}::{time.strftime('%Y%m%d_%H%M%S')}::{uuid.uuid4().hex[:6]}"
        add_chat_session_for_collection(web_collection, new_id)
        sessions = get_chat_sessions_for_collection(web_collection)
        st.session_state["web_active_session"] = new_id

    st.session_state.setdefault("web_active_session", sessions[0])

    pending = st.session_state.pop("web_active_session_pending", None)
    if pending is not None:
        st.session_state["web_active_session"] = pending

    csel, cbtn = st.columns([4, 1], gap="small")
    with csel:
        active_session = st.selectbox(
            "Select a chat",
            options=sessions,
            index=sessions.index(st.session_state["web_active_session"]) if st.session_state["web_active_session"] in sessions else 0,
            key="web_active_session",
            format_func=_kb_chat_label,   # reuse your existing label formatter
            label_visibility="collapsed",
        )

    with cbtn:
        if st.button("➕ New chat", use_container_width=True):
            import uuid
            new_id = f"{session_key('web_chat', web_collection)}::{time.strftime('%Y%m%d_%H%M%S')}::{uuid.uuid4().hex[:6]}"
            add_chat_session_for_collection(web_collection, new_id)
            st.session_state["web_active_session_pending"] = new_id
            st.rerun()

    st.divider()

    # Session-scoped history
    web_sid = f"web::{active_session}"          # stored under history_store["other"]
    hist = get_or_create_history(web_sid)

    # (optional) one-time migrate old web_messages (if it exists) into hist, then stop using it
    if "web_messages" in st.session_state and not getattr(hist, "messages", []):
        for m in st.session_state.web_messages:
            if m.get("role") == "user":
                hist.add_user_message(m.get("content", ""))
            else:
                hist.add_ai_message(m.get("content", ""))
        del st.session_state.web_messages

    if not getattr(hist, "messages", []):
        st.chat_message("assistant").write(
            "Ask me anything. I’ll search Web/Wikipedia/arXiv and answer with sources."
        )

    for msg in getattr(hist, "messages", []):
        msg_type = getattr(msg, "type", "") or msg.__class__.__name__.lower()
        role = "user" if "human" in msg_type else "assistant"
        st.chat_message(role).write(getattr(msg, "content", str(msg)))

    prompt = st.chat_input("Ask a question...", key=f"web_chat_input::{web_sid}")

    def _invoke_text(llm, text: str) -> str:
        try:
            res = llm.invoke(text)
            return getattr(res, "content", str(res)).strip()
        except Exception:
            return text.strip()

    def _standalone_query(llm, hist, user_q: str, max_turns: int = 4) -> str:
        # minimal, safe rewrite: improves follow-ups without bloating the search query too much
        msgs = getattr(hist, "messages", [])[-(max_turns * 2):]
        transcript = []
        for m in msgs:
            t = getattr(m, "type", "") or m.__class__.__name__.lower()
            r = "User" if "human" in t else "Assistant"
            c = (getattr(m, "content", "") or "").strip()
            if c:
                transcript.append(f"{r}: {c}")
        ctx = "\n".join(transcript[-8:])
        rewrite = (
            "Rewrite the user's latest question into a standalone web search query.\n"
            "Keep it short.\n\n"
            f"Conversation:\n{ctx}\n\n"
            f"Latest question: {user_q}\n\n"
            "Standalone search query:"
        )
        out = _invoke_text(llm, rewrite)
        return out or user_q

    if prompt:
        p = prompt.strip().lower()
        meta_q = any(x in p for x in ["what is my first question", "my first question", "what did i ask first", "first thing i asked"])
        if meta_q:
            first_user = None
            for m in getattr(hist, "messages", []):
                t = getattr(m, "type", "") or m.__class__.__name__.lower()
                if "human" in t:
                    first_user = getattr(m, "content", "")
                    break

            st.chat_message("user").write(prompt)
            answer = first_user or "I don't see any earlier user questions in this chat yet."
            hist.add_user_message(prompt)
            hist.add_ai_message(answer)
            st.chat_message("assistant").write(answer)
            st.stop()

        st.chat_message("user").write(prompt)
        hist.add_user_message(prompt)

        try:
            llm = get_llm(settings, model=llm_model, streaming=False)
            q = _standalone_query(llm, hist, prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching (timeboxed) + answering..."):
                    results = search_web_wiki_arxiv(
                        q,
                        ddg_k=5,
                        arxiv_k=3,
                        per_tool_timeout_s=6.0,
                        overall_timeout_s=7.0,
                    )
                    out = answer_with_sources(llm, prompt, results)

                if out.get("used_fallback"):
                    st.caption("⚠️ Sources were unavailable — answered from general LLM knowledge.")
                else:
                    st.caption("✅ Answer grounded in sources (see Sources panel)")

                st.write(out["answer"])

            hist.add_ai_message(out["answer"])

            with st.expander("Sources (tool outputs)"):
                for r in out["sources"]:
                    status = "✅" if r.ok else "❌"
                    meta = f"{status} {r.source}"
                    if r.elapsed_ms is not None:
                        meta += f" ({r.elapsed_ms}ms)"
                    st.markdown(f"### {meta}")
                    st.write(r.content)

        except Exception as e:
            st.error(str(e))

elif page == "Summarizer":
    st.title("📝 URL / YouTube Summarizer")
    st.write("Paste a YouTube or website URL, and get a concise summary.")

    # --- Session control (right side) ---
    sum_collection = "__summarizer__"
    sessions = get_chat_sessions_for_collection(sum_collection)
    if not sessions:
        import uuid
        new_id = f"{session_key('sum_chat', sum_collection)}::{time.strftime('%Y%m%d_%H%M%S')}::{uuid.uuid4().hex[:6]}"
        add_chat_session_for_collection(sum_collection, new_id)
        sessions = get_chat_sessions_for_collection(sum_collection)
        st.session_state["sum_active_session"] = new_id

    st.session_state.setdefault("sum_active_session", sessions[0])
    pending = st.session_state.pop("sum_active_session_pending", None)
    if pending is not None:
        st.session_state["sum_active_session"] = pending

    left, right = st.columns([3, 1], gap="large")

    with right:
        st.caption("Session")
        active_session = st.selectbox(
            "Select a session",
            options=sessions,
            index=sessions.index(st.session_state["sum_active_session"]) if st.session_state["sum_active_session"] in sessions else 0,
            key="sum_active_session",
            format_func=_kb_chat_label,
            label_visibility="collapsed",
        )
        if st.button("➕ New chat", use_container_width=True):
            import uuid
            new_id = f"{session_key('sum_chat', sum_collection)}::{time.strftime('%Y%m%d_%H%M%S')}::{uuid.uuid4().hex[:6]}"
            add_chat_session_for_collection(sum_collection, new_id)
            st.session_state["sum_active_session_pending"] = new_id
            st.rerun()

    # store summaries as (URL -> summary) pairs in history
    sum_sid = f"sum::{active_session}"
    hist = get_or_create_history(sum_sid)

    # --- Existing UI stays on the left ---
    with left:
        url = st.text_input("URL (YouTube or website)", key=f"summarizer_url::{active_session}")
        if st.button("Summarize", type="primary", disabled=not url):
            try:
                llm = get_llm(settings, model=llm_model, streaming=False)
                with st.spinner("Summarizing..."):
                    summary = summarize_url(llm, url)
                hist.add_user_message(url)
                hist.add_ai_message(summary)
            except Exception as e:
                st.error(str(e))

        # Show latest summary (no follow-up chat input)
        msgs = list(getattr(hist, "messages", []))
        pairs = []
        i = 0
        while i + 1 < len(msgs):
            u = getattr(msgs[i], "content", "")
            a = getattr(msgs[i + 1], "content", "")
            if u and a:
                pairs.append((u, a))
            i += 2

        if pairs:
            last_url, last_summary = pairs[-1]
            st.subheader("Summary")
            st.markdown(f"**URL:** [{last_url}]({last_url})")
            st.write(last_summary)

elif page == "RAG Evaluation":
    st.title("📈 RAG Evaluation")
    st.write("Evaluate retriever performance (Hit@K, MRR) using your current Retriever Mode / Reranker settings.")

    from ragmm.eval import load_eval_examples_from_json, evaluate_retriever

    cols_eval = available_collections(settings)  # include kb_default + uploads
    if not cols_eval:
        st.info("No collections found. Build the KB or upload PDFs first.")
    else:
        eval_collection = st.selectbox(
            "Choose a document set to evaluate",
            options=cols_eval,
            key="eval_docset_selected",
        )

        st.write("**Evaluating collection:**")
        st.code(eval_collection)

        k = st.slider("Eval K", 3, 12, value=min(5, settings.top_k), step=1)
        st.caption("Tip: Use gold_chunk_ids for stronger evaluation. Otherwise provide an 'expected' substring.")

        sample = """[
            {"question":"What is retrieval-augmented generation?", "expected":"retrieval"},
            {"question":"Explain BM25 in one sentence.", "expected":"bm25"}
        ]"""
        txt = st.text_area("Paste eval JSON or JSONL", value=sample, height=180, key="eval_text")

        if st.button("Run Evaluation", type="primary", key="run_eval"):
            try:
                examples = load_eval_examples_from_json(txt)
                if not examples:
                    st.warning("No examples found.")
                else:
                    retriever = make_retriever(eval_collection)
                    report = evaluate_retriever(retriever, examples, k=k)

                    st.metric("Hit@K", f"{report['hit_at_k']:.2%}")
                    st.metric("MRR", f"{report['mrr']:.3f}")

                    st.subheader("Per-question results")
                    for i, row in enumerate(report["rows"], start=1):
                        with st.expander(f"#{i} {'✅' if row['hit'] else '❌'}  {row['question'][:80]}"):
                            st.write("**Hit:**", row["hit"])
                            st.write("**Reciprocal rank:**", row["rr"])
                            st.write("**Top sources:**", row["top_sources"])

            except Exception as e:
                st.error(str(e))

