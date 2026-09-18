FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=0

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (no CUDA driver needed — GPU detection is handled in app code)
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Clone DECA + patch legacy API issues
RUN git clone --depth 1 https://github.com/yfeng95/DECA.git /app/DECA && \
    cp -r /app/DECA/data /app/DECA/data_repo && \
    sed -i 's/LandmarksType._2D/LandmarksType.TWO_D/g' /app/DECA/decalib/datasets/detectors.py && \
    sed -i 's/LandmarksType._3D/LandmarksType.THREE_D/g' /app/DECA/decalib/datasets/detectors.py && \
    sed -i "s/checkpoint = torch.load(model_path)/checkpoint = torch.load(model_path, map_location='cpu')/g" /app/DECA/decalib/deca.py

# pytorch3d from tarball (no git clone needed)
RUN pip install --no-cache-dir --no-build-isolation \
    "pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.6.tar.gz"

# Dependencies
RUN pip install --no-cache-dir \
    "numpy<1.24" scipy scikit-image opencv-python-headless \
    PyYAML yacs "kornia>=0.6" face-alignment fvcore iopath ninja \
    chumpy \
    fastapi "uvicorn[standard]" pydantic \
    trimesh pygltflib pillow

ENV PYTHONPATH="/app/DECA:/app/src:${PYTHONPATH}"

# Permissions for DECA data dir
RUN chmod -R 777 /app/DECA/data

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# App code
COPY --chown=user src /app/src
COPY --chown=user entrypoint.sh /app/entrypoint.sh

# Models
COPY --chown=user models /app/models

EXPOSE 7860

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
CMD ["uvicorn", "deca_service.main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "300"]
