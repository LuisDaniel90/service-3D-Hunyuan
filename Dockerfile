FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies (libGL for pymeshlab, libxcb for opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential python3.10 python3-pip python3.10-dev \
    libgl1 libglib2.0-0 libxcb1 libxcb-xinerama0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.5+ (supports Blackwell sm_120)
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# Clone Hunyuan3D-2
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app/Hunyuan3D-2

# Pin numpy<2 BEFORE installing other deps
RUN pip install --no-cache-dir "numpy<2"

# Install Hunyuan3D-2 deps + package
RUN cd /app/Hunyuan3D-2 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Force numpy<2 again (in case requirements.txt overrode it)
RUN pip install --no-cache-dir "numpy<2"

# Install prebuilt custom rasterizer
RUN pip install --no-cache-dir \
    "https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/resolve/main/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"

# Use opencv-headless (no GUI deps needed)
RUN pip install --no-cache-dir opencv-python-headless && \
    pip uninstall -y opencv-python 2>/dev/null; true

# RunPod SDK (install last)
RUN pip install --no-cache-dir runpod~=1.7

COPY handler.py /app/handler.py

# Pre-download model weights
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0/*', 'hunyuan3d-paint-v2-0/*'])"

CMD ["python", "-u", "/app/handler.py"]
