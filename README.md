# Jewellery Visual Search

A fully local, production-ready AI jewellery visual search system.  
Upload any jewellery image and instantly find visually similar designs.  
**No cloud. No Docker. No external services.**

---

## How it works

```
User uploads image
        │
        ▼
  OpenCLIP ViT-B-32
  (generate 512-d embedding)
        │
        ▼
  Qdrant embedded (cosine ANN search)
  → top-K nearest vectors
        │
        ▼
  SQLite metadata lookup
  (filename, path, upload time)
        │
        ▼
  Return ranked image URLs
  with similarity scores
```

---

## Architecture

### Why a hybrid database?

This system uses **two complementary databases** running entirely on your local machine:

| Database | Role |
|----------|------|
| **Qdrant embedded** (vector) | Stores 512-d OpenCLIP embeddings. Performs fast cosine ANN search. No server or Docker needed — runs inside the Python process and persists to `backend/qdrant_storage/`. |
| **SQLite** (relational) | Stores image metadata: filename, local path, upload timestamp, vector ID, status. Used for record management, deletion, and constructing image URLs. |

**Qdrant** finds *which* images are similar; **SQLite** tells you *where* they live on disk.  They are linked by a shared integer ID (SQLite row ID = Qdrant point ID).

### Embedded Qdrant — no Docker required

`qdrant-client` ships with a built-in embedded mode:

```python
client = QdrantClient(path="./qdrant_storage")
```

This runs Qdrant in-process.  Data is persisted to the `qdrant_storage/` folder automatically.  No installation, no server, no Docker.

### Vector strategy

- Single collection: `jewellery_search`
- All images in one vector space — no category separation
- Distance: **Cosine similarity**
- Model: **OpenCLIP ViT-B-32** (`laion2b_s34b_b79k`)
- Dimension: **512**
- All embeddings L2-normalised before storage

### Image storage

All images are stored flat in one folder:

```
backend/dataset/images/
```

No sub-folders. No category structure. Dataset images: `1.jpg`, `2.jpg` …  
User uploads: `upload_<uuid>.jpg`

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| AI model | OpenCLIP ViT-B-32 (laion2b_s34b_b79k) |
| Vector DB | **Qdrant embedded** (no Docker, in-process) |
| Relational DB | SQLite |
| Backend | FastAPI + Python |
| Frontend | HTML + CSS + Vanilla JavaScript |
| Dataset | sidd707/jewelry-design-dataset (HuggingFace) |

---

## Project structure

```
Jewellry_Matching/
├── backend/
│   ├── main.py              ← FastAPI app (4 endpoints)
│   ├── build_database.py    ← one-time setup: download + index
│   ├── database.py          ← SQLite CRUD helpers
│   ├── models.py            ← OpenCLIP singleton embedder
│   ├── requirements.txt
│   ├── jewellery.db         ← created automatically
│   ├── qdrant_storage/      ← created automatically (embedded Qdrant data)
│   ├── dataset/
│   │   └── images/          ← all images stored here (flat)
│   ├── uploads/
│   └── static/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## Installation & setup

### Requirements

- Python 3.10+
- **No Docker needed**
- 4 GB RAM minimum (8 GB recommended for large datasets)
- ~3 GB disk space (model weights + dataset + Qdrant storage)

---

### 1. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> First run downloads OpenCLIP model weights (~350 MB) and PyTorch dependencies.

---

### 2. Build the database

```bash
cd backend
python build_database.py
```

This single command does everything:

1. Creates `dataset/images/`, `qdrant_storage/`, and other directories
2. Creates the SQLite schema (`jewellery.db`)
3. Starts the **embedded Qdrant** — no Docker, no server
4. Creates the `jewellery_search` collection
5. Downloads `sidd707/jewelry-design-dataset` from HuggingFace
6. Flattens all images into `dataset/images/`
7. Generates OpenCLIP embeddings in batches of 32
8. Inserts embeddings into Qdrant
9. Inserts metadata into SQLite

Takes **5–30 minutes** depending on hardware and internet speed.  
**Idempotent** — safe to re-run; already-indexed images are skipped.

---

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The embedded Qdrant starts automatically inside the FastAPI process.  
API available at `http://localhost:8000`.

---

### 4. Open the frontend

Open `frontend/index.html` directly in your browser — no web server needed:

```bash
# Windows
start frontend\index.html

# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html
```

---

## API reference

### `POST /search`

Find visually similar jewellery images.

**Form data:**
- `file` (image) — the query image
- `top_k` (int, optional, default 20, max 100)

**Response:**
```json
[
  {
    "id": 42,
    "image_url": "http://localhost:8000/images/42.jpg",
    "score": 0.9471
  }
]
```

Score is cosine similarity [0, 1]. Higher = more similar.

---

### `POST /add-image`

Add a new image. Instantly searchable after the call returns.

**Form data:** `file` (image)

**Response:**
```json
{
  "id": 1337,
  "filename": "upload_a3f9b2c1.jpg",
  "image_url": "http://localhost:8000/images/upload_a3f9b2c1.jpg",
  "message": "Image added and indexed successfully."
}
```

---

### `DELETE /delete-image/{id}`

Remove from Qdrant, SQLite, and local disk.

```json
{ "id": 1337, "message": "Image deleted successfully." }
```

---

### `GET /images/{filename}`

Serve a local image. Called automatically by the frontend.

---

### `GET /gallery?limit=50&offset=0`

Paginated list of all indexed images, newest first.

---

### `GET /health`

```json
{
  "status": "ok",
  "indexed": 4821,
  "qdrant": "ok (embedded, 4821 vectors)"
}
```

---

## Interactive API docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Performance

| Operation | Typical speed (CPU) |
|-----------|-------------------|
| Embedding — single image | 20–120 ms |
| Qdrant ANN search (top-20, 5k vectors) | < 10 ms |
| End-to-end search | 50–200 ms |
| Batch indexing | ~20–32 images/sec |

---

## Troubleshooting

**Slow on first search**  
OpenCLIP loads into memory on the first request. All subsequent searches are fast.

**Out of memory during `build_database.py`**  
Lower `BATCH_SIZE` in `build_database.py` from `32` to `8` or `16`.

**Dataset download fails**  
```bash
huggingface-cli login
```

**"Cannot reach the backend"**  
```bash
cd backend && uvicorn main:app --reload --port 8000
```

**`qdrant_storage/` is corrupted**  
Delete the folder and re-run `build_database.py` to rebuild from scratch:
```bash
rm -rf backend/qdrant_storage
python build_database.py
```

---

## Design principles

- **Pure image-to-image** — no labels, no classification, no categories
- **Single vector space** — all jewellery in one Qdrant collection
- **Embedded Qdrant** — runs in-process, no Docker, fully local
- **Singleton model** — OpenCLIP loads once per process, reused for every request
- **Realtime indexing** — new images searchable immediately after `/add-image`
- **Corrupted image handling** — bad files skipped automatically during build
- **Path traversal protection** — `/images/{filename}` strips directory components
- **Idempotent build** — `build_database.py` is safe to re-run at any time
