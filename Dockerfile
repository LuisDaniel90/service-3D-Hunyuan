FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Hunyuan3D-2 from GitHub
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app/Hunyuan3D-2 && \
    pip install --no-cache-dir -e /app/Hunyuan3D-2

# RunPod SDK
RUN pip install --no-cache-dir runpod~=1.7

COPY handler.py /app/handler.py

# Pre-download model weights at build time (baked into image = no cold-start download)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0/*', 'hunyuan3d-paint-v2-0/*'])"

CMD ["python", "-u", "/app/handler.py"]
