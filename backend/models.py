"""
models.py
---------
Singleton OpenCLIP embedder.  Import `embedder` anywhere; the model
is loaded exactly once per process.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import open_clip
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512          # ViT-B-32 output dimension


# ---------------------------------------------------------------------------
# Singleton embedder
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """Thread-safe singleton wrapper around OpenCLIP ViT-B-32."""

    _instance: "CLIPEmbedder | None" = None

    def __new__(cls) -> "CLIPEmbedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[OpenCLIP] Loading {CLIP_MODEL} ({CLIP_PRETRAINED}) on {self.device} ...")
        print("[OpenCLIP] First run will download model weights (~350 MB) …")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL,
            pretrained=CLIP_PRETRAINED,
            device=self.device,
        )
        self.model.eval()
        self._initialized = True
        print("[OpenCLIP] Model ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pil(
        self, source: Union[str, Path, Image.Image, bytes]
    ) -> Image.Image:
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        raise TypeError(f"Unsupported image source: {type(source)}")

    def _to_tensor(
        self, source: Union[str, Path, Image.Image, bytes]
    ) -> torch.Tensor:
        return self.preprocess(self._load_pil(source)).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self, image: Union[str, Path, Image.Image, bytes]
    ) -> np.ndarray:
        """
        Generate a normalised 512-d float32 embedding for a single image.
        Raises on unreadable input.
        """
        tensor = self._to_tensor(image)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        vec = features.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def embed_batch(
        self,
        images: List[Union[str, Path, Image.Image, bytes]],
        batch_size: int = 32,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Generate normalised embeddings for a list of images.

        Returns a list of (original_index, embedding) tuples.
        Failed/corrupted images are silently skipped — the caller can
        detect gaps by comparing returned indices against input length.
        """
        results: List[Tuple[int, np.ndarray]] = []

        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            tensors: List[torch.Tensor] = []
            valid_indices: List[int] = []

            for local_i, img in enumerate(chunk):
                try:
                    tensors.append(self._to_tensor(img))
                    valid_indices.append(start + local_i)
                except Exception as exc:
                    print(
                        f"[OpenCLIP] Skipping image at index {start + local_i}: {exc}"
                    )

            if not tensors:
                continue

            batch_tensor = torch.cat(tensors, dim=0)
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)

            embeddings = features.cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

            for local_j, orig_idx in enumerate(valid_indices):
                results.append((orig_idx, embeddings[local_j]))

        return results


# Module-level singleton — import `embedder` wherever you need it
embedder = CLIPEmbedder()
