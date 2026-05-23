"""
build_database.py
-----------------
One-time setup script.  Run this BEFORE starting the FastAPI server.

NO DOCKER REQUIRED.  Qdrant runs embedded inside this process and
stores its data in  backend/qdrant_storage/  on disk.

Download strategy (tried in order):
  1. huggingface_hub snapshot_download  →  extract images from zip / loose files
  2. datasets.load_dataset              →  extract from HF dataset splits
  3. Direct hf_hub_download of any .zip in the repo

Usage:
    cd backend
    python build_database.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# Allow sibling imports when called as __main__
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import database as db
from models import embedder, EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).parent
IMAGES_DIR      = BASE_DIR / "dataset" / "images"
UPLOADS_DIR     = BASE_DIR / "uploads"
STATIC_DIR      = BASE_DIR / "static"
QDRANT_STORAGE  = BASE_DIR / "qdrant_storage"
HF_CACHE_DIR    = BASE_DIR / "hf_cache"       # raw downloaded repo files

COLLECTION      = "jewellery_search"
HF_DATASET      = "sidd707/jewelry-design-dataset"

BATCH_SIZE      = 32
JPEG_QUALITY    = 92
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Directory / Qdrant setup
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (IMAGES_DIR, UPLOADS_DIR, STATIC_DIR, QDRANT_STORAGE, HF_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def connect_qdrant() -> QdrantClient:
    print(f"\n[Qdrant] Starting embedded Qdrant …")
    print(f"[Qdrant] Storage: {QDRANT_STORAGE}")
    client = QdrantClient(path=str(QDRANT_STORAGE))
    print("[Qdrant] Ready.")
    return client


def setup_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"[Qdrant] Collection '{COLLECTION}' already exists — reusing.")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[Qdrant] Created collection '{COLLECTION}' (cosine, {EMBEDDING_DIM}-d).")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _next_counter() -> int:
    """Highest sequential image number already saved in IMAGES_DIR."""
    nums = [int(p.stem) for p in IMAGES_DIR.glob("*.jpg") if p.stem.isdigit()]
    return max(nums, default=0)


def _to_pil(raw) -> Image.Image | None:
    try:
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(raw, (str, Path)):
            return Image.open(raw).convert("RGB")
        if isinstance(raw, dict):
            if raw.get("bytes"):
                return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
            if raw.get("path"):
                return Image.open(raw["path"]).convert("RGB")
    except Exception:
        pass
    return None


def _save_pil(img: Image.Image, dest: Path) -> bool:
    try:
        img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Download method 1: snapshot_download  (primary)
# ---------------------------------------------------------------------------

def _try_snapshot() -> list[Path]:
    """
    Download the raw repo via huggingface_hub.snapshot_download.
    Then extract images from any .zip files found, or copy loose images.
    This bypasses the 'trust_remote_code' restriction entirely.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[Dataset] huggingface_hub not available.")
        return []

    print("[Dataset] Method 1: snapshot_download …")
    try:
        local_dir = snapshot_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            local_dir=str(HF_CACHE_DIR),
            local_dir_use_symlinks=False,   # Windows-safe: no symlinks
        )
        print(f"[Dataset] Repo downloaded to: {local_dir}")
        return _extract_from_dir(Path(local_dir))
    except Exception as exc:
        print(f"[Dataset] snapshot_download failed: {exc}")
        return []


def _extract_from_dir(base: Path) -> list[Path]:
    """Walk a directory: extract images from zips, then copy loose images."""
    saved: list[Path] = []
    counter = _next_counter()

    # ── 1. Extract from zip files ────────────────────────────
    zip_files = sorted(base.rglob("*.zip"))
    if zip_files:
        for zip_path in zip_files:
            print(f"[Dataset] Extracting zip: {zip_path.name}  ({zip_path.stat().st_size // 1_048_576} MB)")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    members = [
                        m for m in zf.namelist()
                        if not m.endswith("/")
                        and Path(m).suffix.lower() in IMAGE_EXTS
                    ]
                    for member in tqdm(members, desc=f"  {zip_path.name}", unit="img"):
                        counter += 1
                        dest = IMAGES_DIR / f"{counter}.jpg"
                        if dest.exists():
                            saved.append(dest)
                            continue
                        try:
                            data = zf.read(member)
                            img = Image.open(io.BytesIO(data)).convert("RGB")
                            if _save_pil(img, dest):
                                saved.append(dest)
                            else:
                                counter -= 1
                        except Exception:
                            counter -= 1
            except zipfile.BadZipFile as exc:
                print(f"[Dataset] Bad zip {zip_path.name}: {exc}")

    # ── 2. Copy loose image files (if no zip found useful images) ──
    if not saved:
        loose = [
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if loose:
            print(f"[Dataset] Found {len(loose)} loose images.")
            for img_path in tqdm(sorted(loose), desc="  Copying images", unit="img"):
                counter += 1
                dest = IMAGES_DIR / f"{counter}.jpg"
                if dest.exists():
                    saved.append(dest)
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    if _save_pil(img, dest):
                        saved.append(dest)
                    else:
                        counter -= 1
                except Exception:
                    counter -= 1

    print(f"[Dataset] Extracted {len(saved)} images from repo.")
    return saved


# ---------------------------------------------------------------------------
# Download method 2: datasets.load_dataset  (fallback)
# ---------------------------------------------------------------------------

def _try_load_dataset() -> list[Path]:
    """Use the HuggingFace datasets library (no trust_remote_code)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    print("[Dataset] Method 2: load_dataset …")
    try:
        ds = load_dataset(HF_DATASET)
    except Exception as exc:
        print(f"[Dataset] load_dataset failed: {exc}")
        return []

    saved: list[Path] = []
    counter = _next_counter()

    for split_name, split_data in ds.items():
        print(f"[Dataset] Split '{split_name}': {len(split_data)} items")

        # Detect image column
        sample = split_data[0]
        image_col: str | None = None
        for col in ["image", "img", "photo", "picture", "Image", "Photo"]:
            if col in sample:
                image_col = col
                break
        if image_col is None:
            for col, val in sample.items():
                if isinstance(val, Image.Image):
                    image_col = col
                    break
        if image_col is None:
            print(f"[Dataset] No image column in '{split_name}' — skipping.")
            continue

        for item in tqdm(split_data, desc=f"  {split_name}", unit="img"):
            raw = item.get(image_col)
            if raw is None:
                continue
            counter += 1
            dest = IMAGES_DIR / f"{counter}.jpg"
            if dest.exists():
                saved.append(dest)
                continue
            pil_img = _to_pil(raw)
            if pil_img is None:
                counter -= 1
                continue
            if _save_pil(pil_img, dest):
                saved.append(dest)
            else:
                counter -= 1

    print(f"[Dataset] Extracted {len(saved)} images via load_dataset.")
    return saved


# ---------------------------------------------------------------------------
# Download method 3: direct hf_hub_download of any zip in repo  (last resort)
# ---------------------------------------------------------------------------

def _try_direct_zip() -> list[Path]:
    """List repo files, download every .zip directly, extract images."""
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        return []

    print("[Dataset] Method 3: direct zip download …")
    try:
        all_files = list(list_repo_files(HF_DATASET, repo_type="dataset"))
    except Exception as exc:
        print(f"[Dataset] Could not list repo files: {exc}")
        return []

    zip_names = [f for f in all_files if f.lower().endswith(".zip")]
    if not zip_names:
        print("[Dataset] No zip files found in repo.")
        return []

    saved: list[Path] = []
    for zip_name in zip_names:
        print(f"[Dataset] Downloading {zip_name} …")
        try:
            local_zip = hf_hub_download(
                repo_id=HF_DATASET,
                filename=zip_name,
                repo_type="dataset",
                local_dir=str(HF_CACHE_DIR),
                local_dir_use_symlinks=False,
            )
            saved.extend(_extract_from_dir(Path(local_zip).parent))
        except Exception as exc:
            print(f"[Dataset] Failed to download {zip_name}: {exc}")

    return saved


# ---------------------------------------------------------------------------
# Master download orchestrator
# ---------------------------------------------------------------------------

def download_and_flatten() -> list[Path]:
    print(f"\n[Dataset] Fetching '{HF_DATASET}' …")

    # If we already have images on disk from a previous run, return them
    existing = sorted(
        [p for p in IMAGES_DIR.glob("*.jpg") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    if existing:
        print(f"[Dataset] Found {len(existing)} images already on disk — skipping download.")
        return existing

    for method in (_try_snapshot, _try_load_dataset, _try_direct_zip):
        result = method()
        if result:
            return result

    print("\n[Dataset] ERROR: All download methods failed.")
    print("          Please download images manually and place them in:")
    print(f"          {IMAGES_DIR}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Embedding + indexing
# ---------------------------------------------------------------------------

def _validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def index_images(client: QdrantClient, image_paths: list[Path]) -> None:
    to_index = [p for p in image_paths if not db.image_exists(p.name)]
    already  = len(image_paths) - len(to_index)
    print(f"\n[Index] {len(to_index)} new images to index  ({already} already in DB).")

    if not to_index:
        print("[Index] Nothing to do.")
        return

    total_indexed = 0
    total_skipped = 0

    for batch_start in tqdm(
        range(0, len(to_index), BATCH_SIZE), desc="  Indexing", unit="batch"
    ):
        chunk = to_index[batch_start : batch_start + BATCH_SIZE]

        valid_paths: list[Path] = []
        row_ids:     list[int]  = []

        for path in chunk:
            if not _validate_image(path):
                print(f"\n[Index] Corrupt — skipping: {path.name}")
                total_skipped += 1
                continue
            row_id = db.insert_image(
                filename=path.name,
                image_path=str(path),
                vector_id=None,
                embedding_status="pending",
            )
            valid_paths.append(path)
            row_ids.append(row_id)

        if not valid_paths:
            continue

        embed_results = embedder.embed_batch(valid_paths, batch_size=BATCH_SIZE)
        if not embed_results:
            total_skipped += len(valid_paths)
            continue

        points: list[PointStruct] = []
        for local_idx, embedding in embed_results:
            row_id   = row_ids[local_idx]
            filename = valid_paths[local_idx].name
            db.update_vector_id(row_id, vector_id=row_id, status="indexed")
            points.append(
                PointStruct(
                    id=row_id,
                    vector=embedding.tolist(),
                    payload={"image_id": row_id, "filename": filename},
                )
            )

        if points:
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            total_indexed += len(points)

    print(f"\n[Index] Done.  Indexed: {total_indexed}.  Skipped: {total_skipped}."
          f"  Total in DB: {db.count_images()}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Jewellery Visual Search — Database Builder")
    print("=" * 60)

    ensure_dirs()
    db.create_tables()

    client = connect_qdrant()
    setup_collection(client)

    image_paths = download_and_flatten()

    if not image_paths:
        print("\n[ERROR] No images found.")
        sys.exit(1)

    index_images(client, image_paths)

    print("\n" + "=" * 60)
    print("  Database ready!")
    print("  Start the backend:")
    print("  uvicorn main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
