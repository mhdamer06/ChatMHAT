# ChatMHAT — Local RAG Chat

A fully local Retrieval-Augmented Generation (RAG) application. Upload documents, ask questions, get streamed answers grounded in your own files — no cloud inference, no external API calls for embeddings or chat.

## Features

- **Hybrid retrieval** — dense vector search (cosine similarity) fused with BM25 keyword search via Reciprocal Rank Fusion (RRF), so both semantic and exact-term matches surface.
- **Local inference** — embedding and chat models run on-device through the Microsoft Foundry Local SDK (CPU, no external API keys required).
- **Resilient embedding pipeline** — adaptive batch splitting automatically halves and retries any batch that fails or times out (Foundry Local's CPU inference has a ~120s ceiling on large batches), so large documents don't fail the whole upload.
- **Real-time upload progress** — Server-Sent Events (SSE) stream batch-by-batch progress from backend to frontend.
- **Streaming chat with sources** — answers stream token-by-token; the documents actually used to answer are shown as source chips underneath the reply.
- **Multi-format ingestion** — PDF, DOCX, TXT, CSV, and XLSX, each with format-appropriate chunking (character-based for prose, row-based for tabular data).
- **Dark, single-file frontend** — no build step; a plain HTML/CSS/JS UI with a pipeline visualization (Embed → Retrieve → Generate) and live document status.

## Architecture

```
Browser (chatmhat.html)
    │  fetch / SSE
    ▼
FastAPI (main.py)
    ├── document_processing.py   # extraction + chunking (PDF/DOCX/TXT/CSV/XLSX)
    ├── retrieval.py              # cosine similarity + BM25 + RRF fusion
    ├── db.py                     # SQLite: documents, chunks, embeddings
    └── Foundry Local SDK         # embedding_client, chat_client (local models)
```

**Flow:**
1. **Upload** → file is chunked → chunks embedded in adaptive batches → saved to SQLite → progress streamed via SSE → retrieval cache (embeddings + BM25 index) reloaded.
2. **Chat** → query embedded → top-20 candidates pulled from both cosine similarity and BM25 → fused with RRF → top-K chunks inserted into the system prompt → chat model streams the answer back over SSE, followed by a `sources` event listing which documents were used.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python |
| Local inference | Microsoft Foundry Local SDK |
| Storage | SQLite |
| Retrieval | NumPy (cosine similarity), `rank_bm25` (BM25Okapi), custom RRF |
| Document parsing | `pypdf`, `python-docx`, `pandas`, `openpyxl` |
| Frontend | Vanilla HTML/CSS/JS (no framework, no build step) |

## Models

| Purpose | Model |
|---|---|
| Embeddings | `qwen3-embedding-0.6b` |
| Chat | `qwen2.5-1.5b` |

Both currently run on CPU — Foundry Local's model catalog only exposed CPU variants for the target GPU, likely a driver/CUDA runtime detection issue; GPU enablement is tracked as future work.

## Setup

### Prerequisites
- Python 3.10+
- [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) installed and configured

### Install
```bash
pip install -r requirements.txt
```

`requirements.txt` should include: `fastapi`, `uvicorn`, `foundry-local-sdk`, `rank_bm25`, `numpy`, `pypdf`, `python-docx`, `pandas`, `openpyxl`.

### Run
```bash
uvicorn main:app --reload
```

On startup, the server downloads (if needed) and loads both the embedding and chat models, initializes the SQLite database (`rag.db`), and rebuilds the retrieval cache from any previously uploaded documents.

Open `index.html` in a browser — it talks to the API at `http://127.0.0.1:8000`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload a document (`multipart/form-data`); returns a `job_id` immediately, processing happens in the background |
| `/upload/progress/{job_id}` | GET | SSE stream of embedding progress for a given job |
| `/documents` | GET | List all indexed documents with chunk counts |
| `/documents/{document_id}` | DELETE | Remove a document and its chunks |
| `/chat` | POST | `{ "query": str, "top_k": int }` → SSE stream of `content` tokens, followed by a `sources` event |

## Supported File Types

`.pdf` `.docx` `.txt` `.csv` `.xlsx`

## Known Limitations / Roadmap

- GPU acceleration not yet working (CPU-only inference currently).
- No authentication or rate limiting in this version — intended for local/trusted-network use.
- Retrieval cache reload isn't fully atomic across concurrent requests (low risk for single-user use).
- Character-based chunking can split mid-word; sentence/semantic-aware chunking is a possible improvement.

## Author

Built by MOHAMAD AMER ALTESHEH — Computer Engineering, İstanbul Kültür University
