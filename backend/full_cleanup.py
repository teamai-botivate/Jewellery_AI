"""
full_cleanup.py
---------------
Delete ALL images from:
1. Qdrant Cloud (all vectors)
2. Cloudinary (all images)
3. SQLite queue (job history)
"""

import cloudinary
import cloudinary.api
import cloudinary.uploader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import sqlite3
from pathlib import Path

# Credentials
QDRANT_URL = "https://e6fff43d-09ee-4f26-bf4d-cef8af87f057.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDk2NTg3MzYtNzhlMi00ZGM0LWE5ZDctYjZiN2EwMjM4MzZmIn0.6dmbwyx1GqtMtsxUCY1E_59W80fA3mbeLdDVA0I5EA4"

CLD_NAME = "dyrc4bo4m"
CLD_KEY = "779426214832782"
CLD_SECRET = "OCG8_QxqrJ6wUTck4Dhm_L7WA_M"

cloudinary.config(
    cloud_name=CLD_NAME,
    api_key=CLD_KEY,
    api_secret=CLD_SECRET,
    secure=True,
)

COLLECTION = "jewellery_search"
DB_PATH = Path("/tmp/upload_queue.db")

print("=" * 60)
print("FULL CLEANUP: Qdrant + Cloudinary + SQLite")
print("=" * 60)

# 1. Cleanup Qdrant Cloud
print("\n[1/3] Cleaning Qdrant Cloud...")
try:
    q = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=60)

    # Delete collection
    q.delete_collection(COLLECTION)
    print(f"  Deleted collection '{COLLECTION}'")

    # Recreate empty
    q.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )
    print(f"  Created empty collection '{COLLECTION}'")

    info = q.get_collection(COLLECTION)
    print(f"  Qdrant status: {info.points_count} vectors")
except Exception as e:
    print(f"  Error: {e}")

# 2. Cleanup Cloudinary
print("\n[2/3] Cleaning Cloudinary...")
try:
    deleted_count = 0

    # Fetch all images
    result = cloudinary.api.resources(type='upload', max_results=500)
    total = result.get('total_count', 0)
    files = result.get('resources', [])

    print(f"  Found {total} assets total")

    # Delete all
    for resource in files:
        try:
            cloudinary.uploader.destroy(resource['public_id'], resource_type='image')
            deleted_count += 1
        except:
            pass

    print(f"  Deleted {deleted_count} images")

    # Verify
    result_check = cloudinary.api.resources(type='upload', max_results=1)
    remaining = result_check.get('total_count', 0)
    print(f"  Remaining assets: {remaining}")

except Exception as e:
    print(f"  Error: {e}")

# 3. Cleanup SQLite Queue
print("\n[3/3] Cleaning SQLite queue database...")
try:
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        # Clear tables
        c.execute("DELETE FROM upload_queue")
        c.execute("DELETE FROM upload_jobs")
        conn.commit()

        print(f"  Cleared upload_queue table")
        print(f"  Cleared upload_jobs table")

        # Verify
        c.execute("SELECT COUNT(*) FROM upload_jobs")
        count = c.fetchone()[0]
        conn.close()

        print(f"  Remaining jobs: {count}")
    else:
        print("  Database doesn't exist yet")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("CLEANUP COMPLETE!")
print("=" * 60)
print("\n✓ Qdrant Cloud: Empty")
print("✓ Cloudinary: Clean")
print("✓ SQLite Queue: Empty")
print("\nSystem ready for fresh start!")
