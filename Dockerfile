# Stage 1: Compile CUDA extensions with devel image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /build

RUN cd /build/hy3dgen/texgen/custom_rasterizer && \
    pip install --no-cache-dir ninja pybind11 && \
    python setup.py bdist_wheel && \
    ls dist/

RUN cd /build/hy3dgen/texgen/differentiable_renderer && \
    python setup.py bdist_wheel && \
    ls dist/

# Stage 2: Runtime image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Pin versions matching the working HF Space
RUN pip install --no-cache-dir \
    "numpy==1.24.4" \
    "diffusers==0.30.0" \
    "transformers==4.46.0" \
    "accelerate==1.1.1" \
    "safetensors==0.4.4" \
    "einops==0.8.0" \
    "omegaconf==2.3.0" \
    "trimesh==4.4.7" \
    "pygltflib==1.16.3" \
    "xatlas==0.0.9" \
    "pymeshlab==2022.2.post3" \
    "opencv-python-headless==4.10.0.84" \
    "rembg==2.0.65" \
    "onnxruntime==1.16.3" \
    "Pillow" \
    "tqdm==4.66.5" \
    "scipy" \
    "ninja==1.11.1.1" \
    "pybind11==2.13.4"

# Clone and install Hunyuan3D-2
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app/Hunyuan3D-2 && \
    cd /app/Hunyuan3D-2 && pip install --no-cache-dir --no-deps -e .

# Install compiled CUDA extensions from builder stage
COPY --from=builder /build/hy3dgen/texgen/custom_rasterizer/dist/*.whl /tmp/
COPY --from=builder /build/hy3dgen/texgen/differentiable_renderer/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# RunPod SDK
RUN pip install --no-cache-dir runpod~=1.7

# Verify ALL imports including texture pipeline
RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('shapegen OK'); \
from hy3dgen.texgen import Hunyuan3DPaintPipeline; print('texgen OK'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK')"

COPY handler.py /app/handler.py

# Pre-download model weights
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0/*', 'hunyuan3d-paint-v2-0/*'])"

CMD ["python", "-u", "/app/handler.py"]
