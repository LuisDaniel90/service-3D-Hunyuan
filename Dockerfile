FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps: Python 3.11, OpenCV libs, pycolmap native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    libgl1 libglib2.0-0 libgomp1 \
    libsm6 libxext6 libxrender1 libice6 libegl1 libgles2 libusb-1.0-0 \
    libboost-filesystem1.74.0 libboost-program-options1.74.0 \
    libboost-graph1.74.0 libboost-regex1.74.0 \
    libflann1.9 libfreeimage3 libmetis5 \
    libsqlite3-0 libcgal-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# COLMAP with CUDA support (dense reconstruction)
RUN pip install --no-cache-dir pycolmap-cuda12

# 3D processing + video frame extraction
RUN pip install --no-cache-dir \
    opencv-python-headless \
    open3d \
    trimesh \
    numpy \
    Pillow \
    scipy

# RunPod SDK
RUN pip install --no-cache-dir runpod~=1.7

# Verify imports
RUN python -c "\
import pycolmap; print(f'pycolmap {pycolmap.__version__}'); \
import cv2; print(f'OpenCV {cv2.__version__}'); \
import open3d as o3d; print(f'Open3D {o3d.__version__}'); \
import trimesh; print('trimesh OK'); \
import numpy; print(f'numpy {numpy.__version__}'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
