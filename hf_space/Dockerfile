FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Pin NumPy < 2 FIRST — torch 2.2.x was compiled against NumPy 1.x
RUN pip install --no-cache-dir "numpy<2"

# Install CPU-only PyTorch (keeps image ~2 GB smaller than CUDA build)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download OpenCLIP model weights into the image so startup is instant
RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); \
print('OpenCLIP weights cached.')"

# Create runtime directories
RUN mkdir -p uploads

# HuggingFace Spaces uses port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
