---
title: Jewellery Visual Search
emoji: 💎
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Jewellery Visual Search

Pure image-to-image AI similarity search for jewellery.  
Upload any jewellery photo → instantly find visually similar designs.

**Powered by:** OpenCLIP ViT-B-32 · Qdrant Cloud · Cloudinary · FastAPI

## Environment Variables (set in Space Settings → Variables and secrets)

| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `CLOUDINARY_CLOUD_NAME` | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | Cloudinary API secret |
