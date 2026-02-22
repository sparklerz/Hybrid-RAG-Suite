---
title: Hybrid RAG Suite
emoji: 📚
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
  - streamlit
  - rag
  - langchain
  - chromadb
  - bm25
  - hybrid-search
  - groq
pinned: false
short_description: A Streamlit RAG app with Dense/BM25/Hybrid retrieval, optional reranking, web+wiki+arXiv chat, summarizer, and evals.
license: agpl-3.0
---

# Hybrid RAG Suite

A multi-page **Streamlit** “RAG app” for experimenting with **dense**, **sparse (BM25)**, and **hybrid** retrieval, with optional reranking and lightweight evaluation utilities.

## What you can do

### 1) Document Chat (Uploads)
- Upload PDFs
- Chunk + embed + index into **Chroma**
- Chat with **conversational RAG** over the uploaded corpus

### 2) Knowledge Base (KB)
- Uses a default KB collection named **`kb_default`**
- Designed for a Thirukkural KB: answers are grounded in retrieved context and can print Tamil couplets

### 3) Web Search Chat (Web + Wikipedia + arXiv)
- Runs **DuckDuckGo search**, **Wikipedia (API)**, and **arXiv (Atom API)** concurrently
- Hard timeouts so the UI stays responsive
- If sources fail, it can fall back to a normal LLM answer (and labels it as fallback)

### 4) Summarizer (URL / YouTube)
- Summarize a webpage or a YouTube video (via transcript)
- Map-reduce summarization (chunk → summarize → combine)
- Note: some environments (including some Spaces) may block YouTube access; the app will show a helpful error in that case

### 5) RAG Evaluation
- Quick metrics for a chosen retriever:
  - **Hit@K**
  - **MRR**
- Supports simple substring expectations or chunk-id based gold labels

### 6) Collection Manager
- List / delete Chroma collections
- Manage associated local artifacts (BM25 cache, docstore)

---

## Retrieval stack (how “Hybrid” works)

- **Dense retrieval:** Chroma vector search (embeddings)
- **Sparse retrieval:** BM25
- **Hybrid retrieval:** Reciprocal Rank Fusion (RRF) of dense + BM25
- **Optional reranker:** Cross-encoder reranking via `sentence-transformers` (CPU-friendly demo)

---

## Hugging Face Spaces (Docker) deployment

This repo is configured for **Docker Spaces** using the .github/workflows/main.yml.

### 1) Create the Space
1. Create a new Space on Hugging Face
2. Choose **SDK → Docker**
3. Push / sync this repository to the Space

### 2) Add required Secret
In your Space: **Settings → Variables and secrets**
- Add **Secret**: `GROQ_API_KEY` (required)

Optional (only if needed by your environment/models):
- `HF_TOKEN`

### 3) (Recommended) Use `/data` for persistence
If you enable persistent storage for the Space, Hugging Face mounts it at `/data`.

Set these as **Variables** (not secrets) to persist indexes/caches:
- `CHROMA_DIR=/data/chroma`
- `ARTIFACTS_DIR=/data/artifacts`

### 4) Important: Startup cleanup behavior
By default, the app **keeps only `kb_default`** across restarts and deletes other collections once per server process.
If you want uploaded collections to persist, change the `PERSISTENT_COLLECTIONS` behavior in `app.py`.

---

## Run locally (Python)

### Prereqs
- Python 3.10+ recommended
- A Groq API key

### Install
```bash
git clone https://github.com/sparklerz/Hybrid-RAG-Suite
cd Hybrid-RAG-Suite

pip install -r requirements.txt
```

### Configure environment
Create a `.env` file:
```bash
GROQ_API_KEY=your_key_here

# Optional
LLM_MODEL=moonshotai/kimi-k2-instruct-0905
EMBED_PROVIDER=huggingface
HF_TOKEN=
CHROMA_DIR=./data/chroma
ARTIFACTS_DIR=./data/artifacts
KB_PDF_DIR=./kb_default
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

### Launch
```bash
streamlit run app.py
```

---

## Configuration reference

| Variable | Default | Purpose |
|---|---:|---|
| `GROQ_API_KEY` | (none) | **Required**. Enables Groq LLM calls |
| `LLM_MODEL` | `moonshotai/kimi-k2-instruct-0905` | Default Groq model |
| `EMBED_PROVIDER` | `huggingface` | Embeddings provider (currently HF only) |
| `HF_TOKEN` | (none) | Optional Hugging Face token |
| `CHROMA_DIR` | `./data/chroma` | Chroma persistent directory |
| `ARTIFACTS_DIR` | `./data/artifacts` | BM25/docstore artifacts directory |
| `KB_PDF_DIR` | `./kb_default` | Folder containing KB PDFs |
| `CHUNK_SIZE` | `1000` | Chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K` | `5` | Retrieval top-k |

---

## Evaluation input format

You can paste either JSON list or JSONL.

### JSON list
```json
[
  {"question": "What is X?", "expected": "some substring"},
  {"question": "Find Y", "gold_chunk_ids": ["abc123", "def456"]}
]
```

### JSONL
```json
{"question":"What is X?","expected":"some substring"}
{"question":"Find Y","gold_chunk_ids":["abc123","def456"]}
```

---

## Notes / gotchas

- **Model downloads:** embeddings and reranker models may download weights at runtime (needs outbound internet).
- **YouTube summarization:** some hosted environments may block YouTube access; prefer webpage summarization if transcripts can’t be retrieved.
- **KB “Thirukkural mode”:** the KB pipeline can extract Kural numbers from retrieved context and fetch Tamil couplets via:
  - `https://tamil-kural-api.vercel.app/api/kural/<n>`
  If the API is unavailable, the app will say Tamil couplet not available.

---

## License

AGPL-3.0 (see `LICENSE`).