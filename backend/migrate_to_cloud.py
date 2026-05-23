"""
migrate_to_cloud.py  (parallel version)
----------------------------------------
Reads all vectors from local Qdrant, uploads images to Cloudinary in parallel,
then batch-upserts into Qdrant Cloud.

Run ONCE locally BEFORE deploying to HuggingFace Spaces:

    cd backend
    python migrate_to_cloud.py

Speedups vs sequential version
  - Pre-fetches all cloud IDs once (no per-point retrieve() calls)
  - Loads all local records into memory first
  - Parallel Cloudinary uploads  (WORKERS threads)
  - Large Qdrant upsert batches  (QDRANT_BATCH = 200)
  - No sleep between uploads
"""

from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

import cloudinary
import cloudinary.uploader
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_CLOUD_KEY = os.getenv("QDRANT_CLOUD_KEY")

CLD_CLOUD_NAME = os.getenv("CLD_CLOUD_NAME")
CLD_API_KEY = os.getenv("CLD_API_KEY")
CLD_API_SECRET = os.getenv("CLD_API_SECRET")

BASE_DIR      = Path(__file__).parent
LOCAL_STORAGE = BASE_DIR / "qdrant_storage"
IMAGES_DIR    = BASE_DIR / "dataset" / "images"
COLLECTION    = "jewellery_search"

WORKERS       = 16   # parallel Cloudinary upload threads
QDRANT_BATCH  = 200  # points per Qdrant upsert call
SCROLL_LIMIT  = 500  # records per scroll page

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cloudinary.config(
    cloud_name=CLD_CLOUD_NAME,
    api_key=CLD_API_KEY,
    api_secret=CLD_API_SECRET,
    secure=True,
)

print("[Local Qdrant ] Opening storage …")
local_qdrant = QdrantClient(path=str(LOCAL_STORAGE))

print("[Cloud Qdrant ] Connecting …")
cloud_qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_CLOUD_KEY, timeout=60)
print("[Cloud Qdrant ] Connected.\n")

# Ensure collection exists in cloud
_existing = {c.name for c in cloud_qdrant.get_collections().collections}
if COLLECTION not in _existing:
    cloud_qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"[Cloud Qdrant ] Created collection '{COLLECTION}'.")
else:
    print(f"[Cloud Qdrant ] Collection '{COLLECTION}' exists — will upsert new points.")

# ---------------------------------------------------------------------------
# Step 1: Pre-fetch all already-migrated IDs from cloud (single pass)
# ---------------------------------------------------------------------------

print("\n[1/3] Fetching already-migrated IDs from cloud …")
cloud_ids: set[int] = set()
offset = None
while True:
    records, next_offset = cloud_qdrant.scroll(
        collection_name=COLLECTION,
        with_vectors=False,
        with_payload=False,
        limit=1000,
        offset=offset,
    )
    for r in records:
        cloud_ids.add(r.id)
    offset = next_offset
    if offset is None:
        break
print(f"    Already in cloud: {len(cloud_ids)}")

# ---------------------------------------------------------------------------
# Step 2: Load ALL local records into memory
# ---------------------------------------------------------------------------

print("\n[2/3] Loading all local records into memory …")
all_local: list = []
offset = None
while True:
    records, next_offset = local_qdrant.scroll(
        collection_name=COLLECTION,
        with_vectors=True,
        with_payload=True,
        limit=SCROLL_LIMIT,
        offset=offset,
    )
    all_local.extend(records)
    offset = next_offset
    if offset is None:
        break

to_migrate = [r for r in all_local if r.id not in cloud_ids]
print(f"    Local total : {len(all_local)}")
print(f"    To migrate  : {len(to_migrate)}")

if not to_migrate:
    print("\n[Done] Cloud already has all vectors. Nothing to migrate.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 3: Parallel Cloudinary upload → batch Qdrant upsert
# ---------------------------------------------------------------------------

def upload_record(record) -> tuple:
    """Upload one image to Cloudinary. Returns (PointStruct, None) or (None, err_str)."""
    filename = (record.payload or {}).get("filename", f"{record.id}.jpg")
    img_path = IMAGES_DIR / filename

    if not img_path.exists():
        return None, f"Image not found: {filename}"

    try:
        img_bytes = img_path.read_bytes()
        result = cloudinary.uploader.upload(
            img_bytes,
            public_id=f"jewellery_search/{record.id}",
            resource_type="image",
            overwrite=True,
            quality="auto",
        )
        point = PointStruct(
            id=record.id,
            vector=record.vector,
            payload={
                "filename":       filename,
                "cloudinary_url": result["secure_url"],
                "public_id":      result["public_id"],
                "image_id":       record.id,
            },
        )
        return point, None
    except Exception as exc:
        return None, str(exc)


print(f"\n[3/3] Uploading {len(to_migrate)} images with {WORKERS} parallel workers …\n")

migrated = 0
skipped  = 0
batch: list[PointStruct] = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(upload_record, r): r for r in to_migrate}

    with tqdm(total=len(to_migrate), unit="img", dynamic_ncols=True) as pbar:
        for future in as_completed(futures):
            point, err = future.result()
            pbar.update(1)

            if err:
                tqdm.write(f"  [WARN] {err}")
                skipped += 1
                continue

            batch.append(point)
            migrated += 1

            if len(batch) >= QDRANT_BATCH:
                cloud_qdrant.upsert(collection_name=COLLECTION, points=batch, wait=True)
                tqdm.write(f"  → Upserted {migrated} vectors so far  |  skipped {skipped}")
                batch.clear()

# Flush any remainder
if batch:
    cloud_qdrant.upsert(collection_name=COLLECTION, points=batch, wait=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

final_info  = cloud_qdrant.get_collection(COLLECTION)
final_count = getattr(final_info, "vectors_count", 0) or 0

print(f"\n{'='*60}")
print(f"  Migration complete!")
print(f"  Migrated this run : {migrated}")
print(f"  Skipped           : {skipped}")
print(f"  Total in cloud    : {final_count}")
print(f"{'='*60}")
print("\n  Next step: set HF Space secrets, then push hf_space/ to HuggingFace.")
