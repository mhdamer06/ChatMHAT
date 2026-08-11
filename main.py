import json
import tempfile
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from foundry_local_sdk import Configuration, FoundryLocalManager
from rank_bm25 import BM25Okapi
import db
from document_processing import extract_and_chunk, find_relevant, bm25_search, reciprocal_rank_fusion

EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b"
CHAT_MODEL_ID = "qwen2.5-1.5b"
TOP_K = 4

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models():
    """Runs once when the server starts, not per-request."""
    db.init_db()

    config = Configuration(app_name="foundry")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ID)
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print(flush=True)
    embedding_model.load()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ID)
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print(flush=True)
    chat_model.load()

    app.state.manager = manager
    app.state.embedding_model = embedding_model
    app.state.chat_model = chat_model
    app.state.embedding_client = embedding_model.get_embedding_client()
    app.state.chat_client = chat_model.get_chat_client()
    app.state.stored_chunks = []

    reload_retrieval_cache()
    print("Models loaded. Ready.", flush=True)


def reload_retrieval_cache():
    with db.get_conn() as conn:
        chunks = db.get_all_chunks(conn)

    app.state.stored_chunks = chunks
    app.state.doc_embeddings = [c["embedding"] for c in chunks]

    if chunks:
        texts = [c["text"] for c in chunks]
        tokenized = [text.lower().split() for text in texts]
        app.state.bm25 = BM25Okapi(tokenized)
    else:
        app.state.bm25 = None


@app.on_event("shutdown")
def unload_models():
    app.state.embedding_model.unload()
    app.state.chat_model.unload()
    print("Models unloaded.", flush=True)


EMBEDDING_BATCH_SIZE = 16 
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_BACKOFF_SEC = 5  

import uuid
import asyncio


app.state.upload_jobs = {}


def _embed_batch_with_retry(batch: list[str]) -> list:
    
    last_error = None
    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            response = app.state.embedding_client.generate_embeddings(batch)
            embeddings = []
            for item in response.data:
                emb = np.asarray(item.embedding, dtype=np.float32)
                emb /= np.linalg.norm(emb)
                embeddings.append(emb)
            return embeddings
        except Exception as e:
            last_error = e
            print(f"[EMBED] Attempt {attempt}/{EMBEDDING_MAX_RETRIES} failed "
                  f"(batch size {len(batch)}): {e}", flush=True)
            if attempt < EMBEDDING_MAX_RETRIES:
                time.sleep(EMBEDDING_RETRY_BACKOFF_SEC * attempt)
    raise last_error


def _embed_with_adaptive_batching(texts: list[str]) -> list:
    
    try:
        return _embed_batch_with_retry(texts)
    except Exception as e:
        if len(texts) <= 1:
            raise
        mid = len(texts) // 2
        print(f"[EMBED] Batch of {len(texts)} failed after retries, "
              f"splitting into {mid} + {len(texts) - mid} and retrying", flush=True)
        left = _embed_with_adaptive_batching(texts[:mid])
        right = _embed_with_adaptive_batching(texts[mid:])
        return left + right


def _embed_and_save_document(filename: str, chunks: list[str], job_id: str | None = None) -> tuple[int, int]:

    total = len(chunks)
    total_batches = (total + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    saved_count = 0

    with db.get_conn() as conn:
        document_id = db.insert_document(conn, filename)
        conn.commit()

        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            batch_num = i // EMBEDDING_BATCH_SIZE + 1
            print(f"[EMBED] Batch {batch_num}/{total_batches} ({len(batch)} chunks)", flush=True)
            t0 = time.time()

            embeddings = _embed_with_adaptive_batching(batch)

            print(f"[EMBED] Batch {batch_num} done in {time.time() - t0:.2f}s", flush=True)

            db.insert_chunks(conn, document_id, batch, embeddings)
            conn.commit()
            saved_count += len(batch)

            if job_id:
                job = app.state.upload_jobs.get(job_id)
                if job:
                    job["completed_batches"] = batch_num
                    job["saved_chunks"] = saved_count

    return document_id, saved_count

async def _run_upload_job(job_id: str, filename: str, chunks: list[str]):
    job = app.state.upload_jobs[job_id]
    try:
        document_id, saved_count = await run_in_threadpool(
            _embed_and_save_document, filename, chunks, job_id
        )
        job["document_id"] = document_id
        job["saved_chunks"] = saved_count
        job["status"] = "done"
        await run_in_threadpool(reload_retrieval_cache)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[UPLOAD] Job {job_id} failed: {e}", flush=True)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    tmp_path = None
    print("1. File uploaded", flush=True)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)

        try:
            chunks = await run_in_threadpool(extract_and_chunk, file.filename, tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks.")

    print("2. Chunks created:", len(chunks), flush=True)

    job_id = str(uuid.uuid4())
    total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    app.state.upload_jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "total_batches": total_batches,
        "completed_batches": 0,
        "saved_chunks": 0,
        "document_id": None,
        "error": None,
    }

    asyncio.create_task(_run_upload_job(job_id, file.filename, chunks))

    # Returned immediately -- the frontend uses job_id to open the progress stream
    return {
        "job_id": job_id,
        "filename": file.filename,
        "chunk_count": len(chunks),
        "total_batches": total_batches,
    }

@app.get("/upload/progress/{job_id}")
async def upload_progress(job_id: str):
    if job_id not in app.state.upload_jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    async def event_stream():
        last_sent = None
        while True:
            job = app.state.upload_jobs.get(job_id)
            if job is None:
                break
            snapshot = json.dumps(job)
            if snapshot != last_sent:
                yield f"data: {snapshot}\n\n"
                last_sent = snapshot
            if job["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/documents")
async def list_documents():
    with db.get_conn() as conn:
        return db.list_documents(conn)


@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    with db.get_conn() as conn:
        db.delete_document(conn, document_id)
    await run_in_threadpool(reload_retrieval_cache)
    return {"message": "Deleted", "document_id": document_id}


class ChatRequest(BaseModel):
    query: str
    top_k: int = TOP_K


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):

    if not req.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    stored_chunks = app.state.stored_chunks

    if not stored_chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload one first."
        )

    
    try:
        query_embedding = _embed_with_adaptive_batching([req.query])[0]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service unavailable: {e}",
        )


    embedding_results = find_relevant(
        query_embedding,
        app.state.doc_embeddings,
        top_k=20
    )

    if app.state.bm25 is not None:
        bm25_results = bm25_search(
            req.query,
            app.state.bm25,
            top_k=20
        )
    else:
        bm25_results = []



    fused = reciprocal_rank_fusion(
        embedding_results,
        bm25_results
    )

    top_matches = [
        {
            **stored_chunks[doc_id],
            "score": score
        }
        for doc_id, score in fused[:req.top_k]
        if 0 <= doc_id < len(stored_chunks)
    ]


    sources = []
    seen_sources = set()

    for chunk in top_matches:
        filename = chunk.get("filename")

        if not filename:
            continue

        if filename not in seen_sources:
            sources.append(
                {
                    "filename": filename,
                    "document_id": chunk["document_id"]
                }
            )

            seen_sources.add(filename)

    context = "\n".join(
        f"- {chunk['text']}"
        for chunk in top_matches
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided context. "
                "If the context doesn't contain enough information, say so.\n\n"
                "Context:\n"
                f"{context}"
            ),
        },
        {
            "role": "user",
            "content": req.query
        },
    ]


    async def event_stream():

        try:

           
            for chunk in app.state.chat_client.complete_streaming_chat(
                messages
            ):

                if await request.is_disconnected():
                    break

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield (
                        f"data: "
                        f"{json.dumps({'content': content})}"
                        f"\n\n"
                    )

            

            yield (
                f"data: "
                f"{json.dumps({'sources': sources})}"
                f"\n\n"
            )

        except Exception as e:

            yield (
                f"data: "
                f"{json.dumps({'error': str(e)})}"
                f"\n\n"
            )

        finally:

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )