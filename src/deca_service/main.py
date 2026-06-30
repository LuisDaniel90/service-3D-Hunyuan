"""FastAPI wrapper for DECA 3D face reconstruction."""

import base64
import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import __version__

# Add DECA to path
DECA_DIR = Path("/app/DECA")
if str(DECA_DIR) not in sys.path:
    sys.path.insert(0, str(DECA_DIR))

logger = logging.getLogger(__name__)

app = FastAPI(title="vision360-deca", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_deca_instance = None


def _get_deca():
    global _deca_instance
    if _deca_instance is not None:
        return _deca_instance

    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    device = "cpu"
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca_cfg.rasterizer_type = "pytorch3d"

    _deca_instance = DECA(config=deca_cfg, device=device)
    return _deca_instance


def _load_image_for_deca(image_bytes: bytes) -> torch.Tensor:
    from decalib.datasets import datasets

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name
    try:
        testdata = datasets.TestData(tmp_path, iscrop=True, face_detector="fan", scale=1.25)
        if len(testdata) == 0:
            raise ValueError("No face detected")
        return testdata[0]["image"].unsqueeze(0)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class PoseView(BaseModel):
    pose: str = Field(..., pattern="^(front|left|right)$")
    image_base64: str


class ReconstructRequest(BaseModel):
    front_image_base64: str
    views: list[PoseView] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "gpu": torch.cuda.is_available()}


@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest):
    try:
        front_bytes = base64.b64decode(req.front_image_base64)
    except Exception:
        raise HTTPException(400, "front_image_base64 invalid")

    deca = _get_deca()
    device = "cpu"

    try:
        image_tensor = _load_image_for_deca(front_bytes).to(device)

        with torch.no_grad():
            codedict = deca.encode(image_tensor)
            opdict, visdict = deca.decode(codedict)

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = str(Path(tmpdir) / "face.obj")
            deca.save_obj(obj_path, opdict)
            mesh = trimesh.load(obj_path, process=False)
            glb = mesh.export(file_type="glb")

    except Exception as e:
        logger.exception("Reconstruction error")
        raise HTTPException(500, str(e))

    return Response(content=glb, media_type="model/gltf-binary")
