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

# Clone Hunyuan3D-2
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app/Hunyuan3D-2

# Pin numpy<2 (PyTorch 2.2 was compiled with numpy 1.x)
# Pin diffusers==0.30.0 (newer versions need torch.xpu which is PyTorch 2.4+)
RUN pip install --no-cache-dir "numpy<2" "diffusers==0.30.0"

# Install Hunyuan3D-2 package + remaining deps
RUN cd /app/Hunyuan3D-2 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Install prebuilt custom rasterizer from HF Space (avoids CUDA compilation)
RUN pip install --no-cache-dir \
    "https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/resolve/main/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"

# RunPod SDK
RUN pip install --no-cache-dir runpod~=1.7

COPY handler.py /app/handler.py

# Pre-download model weights at build time (no cold-start download)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0/*', 'hunyuan3d-paint-v2-0/*'])"

CMD ["python", "-u", "/app/handler.py"]
