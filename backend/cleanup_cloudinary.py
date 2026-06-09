"""
cleanup_cloudinary.py
---------------------
Delete all demo images from Cloudinary folder 'jewellery_search/'
"""

import cloudinary
import cloudinary.api
import cloudinary.uploader
from tqdm import tqdm

cloudinary.config(
    cloud_name="dyrc4bo4m",
    api_key="779426214832782",
    api_secret="OCG8_QxqrJ6wUTck4Dhm_L7WA_M",
    secure=True,
)

print("[1/2] Fetching all images from Cloudinary...")
try:
    resources = cloudinary.api.resources(prefix="jewellery_search/", max_results=500, type="upload")
    total = resources.get("total_count", 0)
    files = resources.get("resources", [])
    print(f"  Found: {total} images")
except Exception as e:
    print(f"  Error: {e}")
    files = []

if files:
    print(f"\n[2/2] Deleting {len(files)} images from Cloudinary...\n")
    deleted = 0
    failed = 0

    for resource in tqdm(files, unit="img"):
        try:
            cloudinary.uploader.destroy(resource["public_id"], resource_type="image")
            deleted += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"  Failed to delete {resource['public_id']}: {e}")

    print(f"\n  Deleted: {deleted}")
    print(f"  Failed: {failed}")
else:
    print("\n  No images to delete.")

print("\n✓ Cloudinary cleanup complete!")
