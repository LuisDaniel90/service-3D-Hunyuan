"""FastAPI wrapper for DECA 3D face reconstruction with multi-view texture."""

import base64
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
from PIL import Image
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


def _compute_view_weight(pose_code: torch.Tensor, pose_label: str) -> torch.Tensor:
    """Compute a UV visibility weight mask based on pose angle.

    Each view sees certain face regions better. We create a soft weight map
    in UV space (256x256) based on the horizontal rotation angle.
    The weight is higher for regions that face the camera directly.
    """
    # Extract global rotation (axis-angle, first 3 of pose vector)
    from pytorch3d.transforms import axis_angle_to_matrix

    rot_aa = pose_code[:, :3]  # (1, 3)
    rot_mat = axis_angle_to_matrix(rot_aa)  # (1, 3, 3)

    # Y-axis rotation angle (horizontal turn) in radians
    # rot_mat[0, 0, 0] = cos(y), rot_mat[0, 2, 0] = sin(y) approximately
    y_angle = torch.atan2(rot_mat[0, 2, 0], rot_mat[0, 0, 0])

    # Create weight based on UV x-coordinate and pose angle
    # UV space: x=0 is right side of face, x=1 is left side
    uv_size = 256
    x_coords = torch.linspace(0, 1, uv_size, device=pose_code.device)

    # Map y_angle to which UV x-region this view sees best
    # front (y≈0) -> center weight, left views (y>0) -> left UV, right (y<0) -> right UV
    center = 0.5 - y_angle.item() / (np.pi * 0.8)  # map angle to UV x center
    center = np.clip(center, 0.1, 0.9)

    # Gaussian weight centered on the region this view sees
    sigma = 0.35  # wide enough for overlap
    weights_x = torch.exp(-((x_coords - center) ** 2) / (2 * sigma**2))

    # Expand to 2D (256x256) - uniform in y direction
    weight_map = weights_x.unsqueeze(0).expand(uv_size, -1)  # (256, 256)

    return weight_map


def _blend_textures(
    deca, codedicts: list[dict], pose_labels: list[str]
) -> torch.Tensor:
    """Decode each view individually and blend their UV textures.

    Returns a blended UV texture tensor (1, 3, 256, 256).
    """
    uv_textures = []
    weight_maps = []

    for i, codedict in enumerate(codedicts):
        with torch.no_grad():
            opdict, _ = deca.decode(codedict)

        # uv_texture_gt is the texture extracted from the input image
        # projected into UV space by DECA. Shape: (1, 3, 256, 256)
        uv_tex = opdict.get("uv_texture_gt")
        if uv_tex is None:
            continue

        uv_textures.append(uv_tex)

        # Compute visibility weight for this view
        weight = _compute_view_weight(codedict["pose"], pose_labels[i])
        weight_maps.append(weight)

    if not uv_textures:
        return None

    # Stack and blend: weighted average across all views
    # textures: list of (1, 3, 256, 256)
    # weights: list of (256, 256)
    blended = torch.zeros_like(uv_textures[0])
    total_weight = torch.zeros(1, 1, 256, 256, device=blended.device)

    for tex, w in zip(uv_textures, weight_maps):
        # Mask out black/empty regions in the texture (no face detected there)
        # If a pixel is near-black, it means DECA couldn't project there
        tex_mask = (tex.sum(dim=1, keepdim=True) > 0.05).float()  # (1, 1, 256, 256)

        w_expanded = w.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
        effective_weight = w_expanded * tex_mask

        blended += tex * effective_weight
        total_weight += effective_weight

    # Avoid division by zero
    total_weight = torch.clamp(total_weight, min=1e-6)
    blended = blended / total_weight

    return blended


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
        pose_labels = []
        front_idx = 0
        for i, (pose, img_bytes) in enumerate(view_images):
            image_tensor = _load_image_for_deca(img_bytes).to(DEVICE)
            with torch.no_grad():
                codedict = deca.encode(image_tensor)
            codedicts.append(codedict)
            pose_labels.append(pose)
            if pose == "front":
                front_idx = i

        # Average shape, expression, and detail across all views
        avg_shape = torch.mean(torch.stack([c["shape"] for c in codedicts]), dim=0)
        avg_exp = torch.mean(torch.stack([c["exp"] for c in codedicts]), dim=0)
        avg_detail = torch.mean(torch.stack([c["detail"] for c in codedicts]), dim=0)

        # Update each codedict with averaged geometry params
        # (keeps individual pose/cam for texture extraction)
        for c in codedicts:
            c["shape"] = avg_shape
            c["exp"] = avg_exp
            c["detail"] = avg_detail

        # Blend UV textures from all views weighted by visibility
        blended_texture = _blend_textures(deca, codedicts, pose_labels)

        # Decode front view for final mesh geometry
        with torch.no_grad():
            opdict, _ = deca.decode(codedicts[front_idx])

        # Export mesh with blended texture
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = str(Path(tmpdir) / "face.obj")

            if blended_texture is not None:
                # Override the texture with our multi-view blend
                opdict["uv_texture_gt"] = blended_texture

            deca.save_obj(obj_path, opdict)

            # Load OBJ with material/texture for GLB export
            obj_dir = Path(tmpdir)
            mesh = trimesh.load(
                obj_path,
                process=False,
                resolver=trimesh.visual.resolvers.FilePathResolver(str(obj_dir)),
            )
            glb = mesh.export(file_type="glb")

    except Exception as e:
        logger.exception("Reconstruction error")
        raise HTTPException(500, str(e))

    return Response(content=glb, media_type="model/gltf-binary")
