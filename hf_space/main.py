"""
main.py — HuggingFace Spaces deployment
----------------------------------------
Uses Qdrant Cloud for vector search and Cloudinary for image storage.
All credentials come from environment variables (set in HF Space Secrets).

Required env vars:
  QDRANT_URL               Qdrant Cloud cluster URL
  QDRANT_API_KEY           Qdrant Cloud API key
  CLOUDINARY_CLOUD_NAME    Cloudinary cloud name
  CLOUDINARY_API_KEY       Cloudinary API key
  CLOUDINARY_API_SECRET    Cloudinary API secret
"""

from __future__ import annotations

import os
import uuid
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

import cloudinary
import cloudinary.uploader
import cloudinary.api
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from models import EMBEDDING_DIM, embedder

# ---------------------------------------------------------------------------
# Config — all from environment variables
# ---------------------------------------------------------------------------

QDRANT_URL      = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY", "").strip()

CLD_CLOUD_NAME  = os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip()
CLD_API_KEY     = os.environ.get("CLOUDINARY_API_KEY", "").strip()
CLD_API_SECRET  = os.environ.get("CLOUDINARY_API_SECRET", "").strip()

COLLECTION      = "jewellery_search"
UPLOAD_FOLDER   = "jewellery_search"   # Cloudinary folder

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL env var is not set. Add it in HF Space → Settings → Variables.")
if not CLD_CLOUD_NAME:
    raise RuntimeError("CLOUDINARY_CLOUD_NAME env var is not set.")

# ---------------------------------------------------------------------------
# SQLite Queue Setup
# ---------------------------------------------------------------------------

DB_PATH = Path("/tmp/upload_queue.db")

def init_db():
    """Initialize SQLite tables for job queue."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS upload_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'processing',
            total_files INTEGER,
            processed_count INTEGER DEFAULT 0,
            failed_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS upload_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            filename TEXT,
            image_bytes BLOB,
            status TEXT DEFAULT 'pending',
            error TEXT,
            qdrant_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(job_id) REFERENCES upload_jobs(job_id)
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------------------------
# Cloudinary setup
# ---------------------------------------------------------------------------

cloudinary.config(
    cloud_name=CLD_CLOUD_NAME,
    api_key=CLD_API_KEY,
    api_secret=CLD_API_SECRET,
    secure=True,
)

# ---------------------------------------------------------------------------
# Qdrant Cloud
# ---------------------------------------------------------------------------

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

_existing = {c.name for c in qdrant.get_collections().collections}
if COLLECTION not in _existing:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[Qdrant] Created collection '{COLLECTION}'.")
else:
    print(f"[Qdrant] Collection '{COLLECTION}' ready.")


# ---------------------------------------------------------------------------
# Qdrant search — handles both old (.search) and new (.query_points) API
# ---------------------------------------------------------------------------

def _qdrant_search(query_vector: list, limit: int) -> list:
    if hasattr(qdrant, "query_points"):
        result = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return result.points
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )


# ---------------------------------------------------------------------------
# Cloudinary helpers
# ---------------------------------------------------------------------------

def _upload_image(image_bytes: bytes, original_filename: str) -> dict:
    """Upload bytes to Cloudinary; returns {url, public_id}."""
    public_id = f"{UPLOAD_FOLDER}/{uuid.uuid4().hex}"
    result = cloudinary.uploader.upload(
        image_bytes,
        public_id=public_id,
        resource_type="image",
        overwrite=False,
        quality="auto",
        fetch_format="auto",
    )
    return {"url": result["secure_url"], "public_id": result["public_id"]}


def _delete_cloudinary(public_id: str) -> None:
    try:
        cloudinary.uploader.destroy(public_id, resource_type="image")
    except Exception as exc:
        print(f"[Cloudinary] Could not delete {public_id}: {exc}")


def _point_id_from(public_id: str) -> int:
    """Stable positive integer ID derived from a Cloudinary public_id."""
    import hashlib
    return int(hashlib.sha256(public_id.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Background Upload Worker
# ---------------------------------------------------------------------------

def _process_queue_item(job_id: str, queue_id: int, filename: str, image_bytes: bytes) -> dict:
    """Process one item from queue. Returns result dict."""
    try:
        # Upload to Cloudinary
        cloud = _upload_image(image_bytes, filename)
        # Generate embedding
        embedding = embedder.embed(image_bytes)
        # Qdrant ID
        point_id = _point_id_from(cloud["public_id"])
        # Upsert to Qdrant
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "filename": filename,
                        "cloudinary_url": cloud["url"],
                        "public_id": cloud["public_id"],
                        "uploaded_at": datetime.utcnow().isoformat(),
                    },
                )
            ],
            wait=True,
        )
        return {"success": True, "qdrant_id": point_id, "error": None}
    except Exception as exc:
        return {"success": False, "qdrant_id": None, "error": str(exc)}


def background_upload_worker():
    """Background thread: processes upload queue continuously."""
    while True:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()

            # Get next batch of pending items
            c.execute("""
                SELECT id, job_id, filename, image_bytes
                FROM upload_queue
                WHERE status='pending'
                LIMIT 50
            """)
            items = c.fetchall()
            conn.close()

            if not items:
                time.sleep(1)
                continue

            # Process each item
            for queue_id, job_id, filename, image_bytes in items:
                result = _process_queue_item(job_id, queue_id, filename, image_bytes)

                # Update queue item
                conn = sqlite3.connect(str(DB_PATH))
                c = conn.cursor()

                if result["success"]:
                    c.execute("""
                        UPDATE upload_queue
                        SET status='done', qdrant_id=?
                        WHERE id=?
                    """, (result["qdrant_id"], queue_id))

                    # Update job stats
                    c.execute("""
                        UPDATE upload_jobs
                        SET processed_count = processed_count + 1
                        WHERE job_id=?
                    """, (job_id,))
                else:
                    c.execute("""
                        UPDATE upload_queue
                        SET status='failed', error=?
                        WHERE id=?
                    """, (result["error"], queue_id))

                    # Update job stats
                    c.execute("""
                        UPDATE upload_jobs
                        SET failed_count = failed_count + 1
                        WHERE job_id=?
                    """, (job_id,))

                conn.commit()
                conn.close()

                time.sleep(0.1)  # Small delay between items

            # Check if job is complete
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute("""
                SELECT job_id, total_files, processed_count, failed_count
                FROM upload_jobs
                WHERE status='processing'
            """)
            jobs = c.fetchall()

            for job_id_check, total, processed, failed in jobs:
                if processed + failed >= total:
                    c.execute("""
                        UPDATE upload_jobs
                        SET status='completed'
                        WHERE job_id=?
                    """, (job_id_check,))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"[Background Worker Error] {e}")
            time.sleep(5)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Jewellery Visual Search",
    description="Image-to-image similarity search — OpenCLIP + Qdrant Cloud + Cloudinary",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}


def _validate(file: UploadFile) -> None:
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported type: {file.content_type}")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    id: int
    image_url: str
    score: float

class AddImageResponse(BaseModel):
    id: int
    filename: str
    image_url: str
    message: str

class DeleteResponse(BaseModel):
    id: int
    message: str

class HealthResponse(BaseModel):
    status: str
    indexed: int
    qdrant: str
    cloudinary: str

class BatchUploadResult(BaseModel):
    id: int
    filename: str
    image_url: str
    error: str | None = None

class BatchUploadResponse(BaseModel):
    success: List[BatchUploadResult]
    failed: List[dict]
    total: int
    message: str

class AsyncUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str

class UploadStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    failed: int
    remaining: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        info    = qdrant.get_collection(COLLECTION)
        count   = getattr(info, "vectors_count", 0) or 0
        q_status = f"ok ({count} vectors)"
    except Exception as exc:
        q_status = f"error: {exc}"
        count    = 0

    try:
        cloudinary.api.ping()
        c_status = "ok"
    except Exception as exc:
        c_status = f"error: {exc}"

    return HealthResponse(status="ok", indexed=count, qdrant=q_status, cloudinary=c_status)


@app.post("/search", response_model=List[SearchResult])
async def search(
    file: UploadFile = File(...),
    top_k: int = Query(20, ge=1, le=100),
):
    """Upload a jewellery image → get top-K visually similar results."""
    _validate(file)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    try:
        query_vec = embedder.embed(image_bytes)
    except Exception as exc:
        raise HTTPException(422, f"Cannot process image: {exc}")

    hits = _qdrant_search(query_vec.tolist(), limit=top_k)

    results: List[SearchResult] = []
    for hit in hits:
        payload     = hit.payload or {}
        cloudinary_url = payload.get("cloudinary_url", "")
        if not cloudinary_url:
            continue

        # Normalise ID to positive int
        raw_id = hit.id
        if isinstance(raw_id, int):
            point_id = raw_id
        else:
            point_id = abs(hash(str(raw_id))) % (10 ** 9)

        results.append(SearchResult(
            id=point_id,
            image_url=cloudinary_url,
            score=round(float(hit.score), 6),
        ))

    return results


@app.post("/add-image", response_model=AddImageResponse)
async def add_image(file: UploadFile = File(...)):
    """Add a new jewellery image — stored on Cloudinary, indexed in Qdrant."""
    _validate(file)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    filename = file.filename or "upload.jpg"

    # 1. Upload to Cloudinary
    try:
        cloud = _upload_image(image_bytes, filename)
    except Exception as exc:
        raise HTTPException(500, f"Cloudinary upload failed: {exc}")

    # 2. Generate embedding
    try:
        embedding = embedder.embed(image_bytes)
    except Exception as exc:
        _delete_cloudinary(cloud["public_id"])
        raise HTTPException(422, f"Cannot embed image: {exc}")

    # 3. Stable integer point ID
    point_id = _point_id_from(cloud["public_id"])

    # 4. Upsert to Qdrant Cloud
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "filename": filename,
                    "cloudinary_url": cloud["url"],
                    "public_id": cloud["public_id"],
                    "uploaded_at": datetime.utcnow().isoformat(),
                },
            )
        ],
        wait=True,
    )

    return AddImageResponse(
        id=point_id,
        filename=filename,
        image_url=cloud["url"],
        message="Uploaded to Cloudinary and indexed in Qdrant.",
    )


@app.delete("/delete-image/{image_id}", response_model=DeleteResponse)
def delete_image(image_id: int):
    """Delete image from Qdrant and Cloudinary."""
    try:
        points = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=[image_id],
            with_payload=True,
        )
    except Exception:
        points = []

    if not points:
        raise HTTPException(404, "Image not found.")

    public_id = (points[0].payload or {}).get("public_id", "")
    if public_id:
        _delete_cloudinary(public_id)

    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=PointIdsList(points=[image_id]),
        wait=True,
    )

    return DeleteResponse(id=image_id, message="Deleted from Cloudinary and Qdrant.")


def _process_single_image(file_data: tuple) -> dict:
    """Process one image: validate, upload, embed, index. Returns result dict."""
    filename, image_bytes = file_data
    try:
        cloud = _upload_image(image_bytes, filename)
        embedding = embedder.embed(image_bytes)
        point_id = _point_id_from(cloud["public_id"])
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "filename": filename,
                        "cloudinary_url": cloud["url"],
                        "public_id": cloud["public_id"],
                        "uploaded_at": datetime.utcnow().isoformat(),
                    },
                )
            ],
            wait=True,
        )
        return {
            "success": True,
            "id": point_id,
            "filename": filename,
            "image_url": cloud["url"],
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "filename": filename,
            "error": str(exc),
        }


@app.post("/add-image-batch-async", response_model=AsyncUploadResponse)
async def add_image_batch_async(files: List[UploadFile] = File(...)):
    """Upload images asynchronously. Returns job_id to track progress."""
    if not files:
        raise HTTPException(400, "No files provided.")

    # Create job
    job_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        INSERT INTO upload_jobs (job_id, status, total_files)
        VALUES (?, 'processing', ?)
    """, (job_id, len(files)))

    # Add files to queue
    for file in files:
        if file.content_type and file.content_type not in ALLOWED_MIME:
            continue
        image_bytes = await file.read()
        if image_bytes:
            c.execute("""
                INSERT INTO upload_queue (job_id, filename, image_bytes, status)
                VALUES (?, ?, ?, 'pending')
            """, (job_id, file.filename or "upload.jpg", image_bytes))

    conn.commit()
    conn.close()

    return AsyncUploadResponse(
        job_id=job_id,
        status="processing",
        message=f"Upload started. {len(files)} files queued for processing.",
    )


@app.get("/upload-status/{job_id}", response_model=UploadStatusResponse)
def get_upload_status(job_id: str):
    """Get status of an upload job."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT status, total_files, processed_count, failed_count
        FROM upload_jobs
        WHERE job_id=?
    """, (job_id,))

    result = c.fetchone()
    conn.close()

    if not result:
        raise HTTPException(404, f"Job {job_id} not found.")

    status, total, processed, failed = result
    remaining = total - processed - failed

    return UploadStatusResponse(
        job_id=job_id,
        status=status,
        total=total,
        processed=processed,
        failed=failed,
        remaining=remaining,
    )


@app.get("/jobs-history")
def get_jobs_history():
    """Get all upload jobs history."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT job_id, status, total_files, processed_count, failed_count, created_at
        FROM upload_jobs
        ORDER BY created_at DESC
        LIMIT 100
    """)

    jobs = c.fetchall()
    conn.close()

    return [
        {
            "job_id": job[0],
            "status": job[1],
            "total": job[2],
            "processed": job[3],
            "failed": job[4],
            "created_at": job[5],
        }
        for job in jobs
    ]


@app.post("/add-image-batch", response_model=BatchUploadResponse)
async def add_image_batch(files: List[UploadFile] = File(...)):
    """Upload multiple images (unlimited). Auto-batches in groups of 100 with 4-thread parallel processing."""
    if not files:
        raise HTTPException(400, "No files provided.")

    file_data_list = []
    for file in files:
        if file.content_type and file.content_type not in ALLOWED_MIME:
            continue
        image_bytes = await file.read()
        if image_bytes:
            file_data_list.append((file.filename or "upload.jpg", image_bytes))

    if not file_data_list:
        raise HTTPException(400, "No valid images in batch.")

    success_list = []
    failed_list = []

    BATCH_SIZE = 100
    total_batches = (len(file_data_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(file_data_list))
        batch = file_data_list[start_idx:end_idx]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_process_single_image, fd) for fd in batch]
            for future in futures:
                result = future.result()
                if result["success"]:
                    success_list.append(
                        BatchUploadResult(
                            id=result["id"],
                            filename=result["filename"],
                            image_url=result["image_url"],
                        )
                    )
                else:
                    failed_list.append(
                        {
                            "filename": result["filename"],
                            "error": result["error"],
                        }
                    )

    return BatchUploadResponse(
        success=success_list,
        failed=failed_list,
        total=len(file_data_list),
        message=f"{len(success_list)} uploaded, {len(failed_list)} failed. (Processed in {total_batches} batch(es))",
    )


# ---------------------------------------------------------------------------
# Startup: Start background worker
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Start background upload worker thread on app startup."""
    worker_thread = threading.Thread(
        target=background_upload_worker,
        daemon=True,
        name="UploadWorker"
    )
    worker_thread.start()
    print("[Background Worker] Started")


# ---------------------------------------------------------------------------
# Serve frontend — MUST be registered last so API routes take priority
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
