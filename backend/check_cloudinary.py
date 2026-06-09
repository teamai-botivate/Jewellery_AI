import cloudinary
import cloudinary.api

cloudinary.config(
    cloud_name="dyrc4bo4m",
    api_key="779426214832782",
    api_secret="OCG8_QxqrJ6wUTck4Dhm_L7WA_M",
    secure=True,
)

print("Checking Cloudinary folder status...")
try:
    result = cloudinary.api.resources(type="upload", prefix="jewellery_search/", max_results=1)
    total = result.get("total_count", 0)
    print(f"Images in jewellery_search/: {total}")

    if total > 0:
        sample = result["resources"][0]
        public_id = sample["public_id"]
        print(f"Sample: {public_id}")
    else:
        print("✓ Folder is CLEAN (0 images)")
except Exception as e:
    print(f"Error: {e}")
