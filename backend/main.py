"""
main.py
-------
FastAPI backend for the Jewellery Visual Search system.

Endpoints:
  POST   /search                 – image-to-image similarity search
  POST   /add-image              – add a new image (instantly searchable)
  DELETE /delete-image/{id}      – remove image from all stores
  GET    /images/{filename}      – serve a local image file
  GET    /gallery                – paginated list of all indexed images
  GET    /health                 – health check + index count
  GET    /                       – serves the frontend UI

Start:
  cd backend
  uvicorn main:app --port 8000
  Then open http://localhost:8000
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

import database as db
from models import EMBEDDING_DIM, embedder

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).parent
IMAGES_DIR      = BASE_DIR / "dataset" / "images"
UPLOADS_DIR     = BASE_DIR / "uploads"
STATIC_DIR      = BASE_DIR / "static"
QDRANT_STORAGE  = BASE_DIR / "qdrant_storage"
FRONTEND_DIR    = BASE_DIR.parent / "frontend"

COLLECTION    = "jewellery_search"
DEFAULT_TOP_K = 20
MAX_TOP_K     = 100

ALLOWED_MIME  = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
ALLOWED_EXT   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# ---------------------------------------------------------------------------
# Startup — directories, SQLite, embedded Qdrant
# ---------------------------------------------------------------------------

for _d in (IMAGES_DIR, UPLOADS_DIR, STATIC_DIR, QDRANT_STORAGE):
    _d.mkdir(parents=True, exist_ok=True)

db.create_tables()

qdrant = QdrantClient(path=str(QDRANT_STORAGE))

_existing = {c.name for c in qdrant.get_collections().collections}
if COLLECTION not in _existing:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ---------------------------------------------------------------------------
# Qdrant search compatibility wrapper
# qdrant-client >= 1.7.4  replaced  .search()  with  .query_points()
# ---------------------------------------------------------------------------

def _qdrant_search(query_vector: list, limit: int) -> list:
    """Return a list of ScoredPoint, compatible with old and new qdrant-client."""
    # New API (>= 1.7.4)
    if hasattr(qdrant, "query_points"):
        result = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return result.points

    # Legacy API (< 1.7.4)
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Jewellery Visual Search API",
    description="Pure image-to-image similarity search — OpenCLIP + Qdrant.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# allow_credentials must be False when allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class GalleryItem(BaseModel):
    id: int
    filename: str
    image_url: str
    uploaded_at: str
    embedding_status: str


class HealthResponse(BaseModel):
    status: str
    indexed: int
    qdrant: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_url(filename: str) -> str:
    return f"http://localhost:8000/images/{filename}"


def _validate_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'. Allowed: {', '.join(ALLOWED_MIME)}",
        )


def _safe_filename(name: str) -> str:
    return Path(name).name


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    try:
        info = qdrant.get_collection(COLLECTION)
        count = getattr(info, "vectors_count", None) or 0
        qdrant_status = f"ok (embedded, {count} vectors)"
    except Exception as exc:
        qdrant_status = f"error: {exc}"
    return HealthResponse(status="ok", indexed=db.count_images(), qdrant=qdrant_status)


@app.post("/search", response_model=List[SearchResult], tags=["Search"])
async def search(
    file: UploadFile = File(...),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
):
    """Upload a jewellery image → get ranked similar images."""
    _validate_upload(file)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        query_vec = embedder.embed(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot process image: {exc}")

    hits = _qdrant_search(query_vec.tolist(), limit=top_k)

    if not hits:
        return []

    hit_ids      = [int(h.id) for h in hits]
    score_by_id  = {int(h.id): round(float(h.score), 6) for h in hits}
    metadata_map = db.get_images_by_ids(hit_ids)

    results: List[SearchResult] = []
    for hit_id in hit_ids:
        meta = metadata_map.get(hit_id)
        if meta is None:
            continue
        results.append(
            SearchResult(
                id=meta["id"],
                image_url=_image_url(meta["filename"]),
                score=score_by_id[hit_id],
            )
        )
    return results


@app.post("/add-image", response_model=AddImageResponse, tags=["Management"])
async def add_image(file: UploadFile = File(...)):
    """Add a new image — saved locally and indexed immediately."""
    _validate_upload(file)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    ext = Path(file.filename or "x.jpg").suffix.lower()
    if ext not in ALLOWED_EXT:
        ext = ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex}{ext}"
    dest_path   = IMAGES_DIR / unique_name

    dest_path.write_bytes(image_bytes)

    try:
        embedding = embedder.embed(image_bytes)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Cannot process image: {exc}")

    row_id = db.insert_image(
        filename=unique_name,
        image_path=str(dest_path),
        vector_id=None,
        embedding_status="pending",
    )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=row_id,
                vector=embedding.tolist(),
                payload={"image_id": row_id, "filename": unique_name},
            )
        ],
        wait=True,
    )

    db.update_vector_id(row_id, vector_id=row_id, status="indexed")

    return AddImageResponse(
        id=row_id,
        filename=unique_name,
        image_url=_image_url(unique_name),
        message="Image added and indexed successfully.",
    )


@app.delete("/delete-image/{image_id}", response_model=DeleteResponse, tags=["Management"])
def delete_image(image_id: int):
    """Remove image from Qdrant, SQLite, and local disk."""
    meta = db.get_image_by_id(image_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    try:
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=PointIdsList(points=[image_id]),
            wait=True,
        )
    except Exception as exc:
        print(f"[Qdrant] Warning — could not delete point {image_id}: {exc}")

    db.delete_image_by_id(image_id)

    local_file = Path(meta["image_path"])
    if local_file.exists():
        local_file.unlink(missing_ok=True)

    return DeleteResponse(id=image_id, message="Image deleted successfully.")


@app.get("/images/{filename}", tags=["Files"])
def serve_image(filename: str):
    """Serve a local image by filename (path-traversal safe)."""
    safe = _safe_filename(filename)
    path = IMAGES_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(path))


@app.get("/gallery", response_model=List[GalleryItem], tags=["Management"])
def gallery(
    limit:  int = Query(50, ge=1, le=200),
    offset: int = Query(0,  ge=0),
):
    """Paginated list of all indexed images, newest first."""
    rows = db.get_all_images(limit=limit, offset=offset)
    return [
        GalleryItem(
            id=r["id"],
            filename=r["filename"],
            image_url=_image_url(r["filename"]),
            uploaded_at=r["uploaded_at"],
            embedding_status=r["embedding_status"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Serve frontend — MUST be last so API routes take priority
# Fixes the CORS null-origin problem: everything runs on http://localhost:8000
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
