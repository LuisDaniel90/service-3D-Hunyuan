"""FastAPI wrapper for DECA 3D face reconstruction."""

import base64
import logging
import sys
import tempfile
from pathlib import Path

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

# Auto-detect device
def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detectada: {name}")
        return "cuda"
    logger.info("Sin GPU — usando CPU")
    return "cpu"

DEVICE = _detect_device()

_deca_instance = None


def _get_deca():
    global _deca_instance
    if _deca_instance is not None:
        return _deca_instance

    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca_cfg.rasterizer_type = "pytorch3d"

    _deca_instance = DECA(config=deca_cfg, device=DEVICE)
    return _deca_instance


def _load_image_for_deca(image_bytes: bytes) -> torch.Tensor:
    """Patch face_alignment to use correct device, then load image."""
    import face_alignment
    from decalib.datasets import detectors

    # Patch FAN detector to use auto-detected device
    class FAN:
        def __init__(self):
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=DEVICE,
            )

        def run(self, image):
            out = self.model.get_landmarks(image)
            if out is None:
                return [0], "kpt68", []
            return [0], "kpt68", [o.squeeze() for o in out]

    # Replace detector with our patched version
    detectors.FAN = FAN

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
    pose: str = Field(..., pattern="^(front|left_45|right_45|left_90|right_90|up)$")
    image_base64: str


class ReconstructRequest(BaseModel):
    views: list[PoseView] = Field(..., min_length=1, max_length=6)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "device": DEVICE,
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest):
    # Decode all view images
    view_images: list[tuple[str, bytes]] = []
    for i, view in enumerate(req.views):
        try:
            img_bytes = base64.b64decode(view.image_base64)
            view_images.append((view.pose, img_bytes))
        except Exception:
            raise HTTPException(400, f"views[{i}].image_base64 invalid")

    deca = _get_deca()

    try:
        # Encode all views through DECA
        codedicts = []
        front_idx = 0
        for i, (pose, img_bytes) in enumerate(view_images):
            image_tensor = _load_image_for_deca(img_bytes).to(DEVICE)
            with torch.no_grad():
                codedict = deca.encode(image_tensor)
            codedicts.append(codedict)
            if pose == "front":
                front_idx = i

        # Average shape params across all successfully encoded views
        # Shape should be consistent across views; averaging reduces noise
        avg_shape = torch.mean(torch.stack([c["shape"] for c in codedicts]), dim=0)

        # Use front view as base, override shape with averaged shape
        final_code = codedicts[front_idx].copy()
        final_code["shape"] = avg_shape

        with torch.no_grad():
            opdict, visdict = deca.decode(final_code)

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = str(Path(tmpdir) / "face.obj")
            deca.save_obj(obj_path, opdict)
            mesh = trimesh.load(obj_path, process=False)
            glb = mesh.export(file_type="glb")

    except Exception as e:
        logger.exception("Reconstruction error")
        raise HTTPException(500, str(e))

    return Response(content=glb, media_type="model/gltf-binary")
