"""FastAPI wrapper for DECA 3D face reconstruction.

Uses UV textures at 1024px resolution with multi-view gap filling
from lateral photos.
"""

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
from pydantic import BaseModel, Field

from . import __version__

DECA_DIR = Path("/app/DECA")
if str(DECA_DIR) not in sys.path:
    sys.path.insert(0, str(DECA_DIR))

logger = logging.getLogger(__name__)

app = FastAPI(title="vision360-deca", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

UV_SIZE = 1024


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
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
    # Keep uv_size=256 (D_detail generator is hardcoded to 256)
    # We upscale the final texture to UV_SIZE after generation
    deca_cfg.rasterizer_type = "pytorch3d"

    _deca_instance = DECA(config=deca_cfg, device=DEVICE)
    return _deca_instance


def _load_image_for_deca(image_bytes: bytes) -> torch.Tensor:
    import face_alignment
    from decalib.datasets import detectors

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
                return [0], "kpt68"
            kpt = out[0].squeeze()
            return [
                int(np.min(kpt[:, 0])), int(np.min(kpt[:, 1])),
                int(np.max(kpt[:, 0])), int(np.max(kpt[:, 1])),
            ], "kpt68"

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


def _fix_glb_material(glb_bytes: bytes) -> bytes:
    """Patch GLB material for full brightness and double-sided rendering."""
    import json, struct

    magic, version, total_len = struct.unpack_from('<III', glb_bytes, 0)
    json_len = struct.unpack_from('<I', glb_bytes, 12)[0]
    json_type = struct.unpack_from('<I', glb_bytes, 16)[0]
    json_data = json.loads(glb_bytes[20:20 + json_len])

    for mat in json_data.get('materials', []):
        pbr = mat.get('pbrMetallicRoughness', {})
        pbr['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
        pbr['metallicFactor'] = 0.0
        pbr['roughnessFactor'] = 0.7
        mat['pbrMetallicRoughness'] = pbr
        mat['doubleSided'] = True

    new_json = json.dumps(json_data, separators=(',', ':')).encode('utf-8')
    padding = (4 - len(new_json) % 4) % 4
    new_json += b' ' * padding

    bin_chunk = glb_bytes[20 + json_len:]
    new_total = 12 + 8 + len(new_json) + len(bin_chunk)

    result = struct.pack('<III', magic, version, new_total)
    result += struct.pack('<II', len(new_json), json_type)
    result += new_json
    result += bin_chunk
    return result


class PoseView(BaseModel):
    pose: str = Field(..., pattern="^(front|left45|right45|left|right|eyes)$")
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
    view_images: list[tuple[str, bytes]] = []
    for i, view in enumerate(req.views):
        try:
            img_bytes = base64.b64decode(view.image_base64)
            view_images.append((view.pose, img_bytes))
        except Exception:
            raise HTTPException(400, f"views[{i}].image_base64 invalid")

    deca = _get_deca()

    try:
        # 1. Encode all views (skip eyes)
        codedicts = []
        pose_labels = []
        front_idx = 0

        for pose, img_bytes in view_images:
            if pose == "eyes":
                continue
            try:
                image_tensor = _load_image_for_deca(img_bytes).to(DEVICE)
                with torch.no_grad():
                    codedict = deca.encode(image_tensor)
                codedicts.append(codedict)
                pose_labels.append(pose)
                if pose == "front":
                    front_idx = len(codedicts) - 1
            except Exception as e:
                logger.warning(f"Skipping view {pose}: {e}")

        if not codedicts:
            raise ValueError("No faces detected in any view")

        # 2. Average shape for better geometry
        avg_shape = torch.mean(torch.stack([c["shape"] for c in codedicts]), dim=0)
        for c in codedicts:
            c["shape"] = avg_shape

        # 3. Decode front view → base texture
        with torch.no_grad():
            front_opdict, _ = deca.decode(codedicts[front_idx])

        base_tex = front_opdict["uv_texture_gt"]  # (1, 3, UV_SIZE, UV_SIZE)
        covered = (base_tex.sum(dim=1, keepdim=True) > 0.1).float()

        # 4. Fill UV gaps with lateral views
        for i, codedict in enumerate(codedicts):
            if i == front_idx:
                continue
            with torch.no_grad():
                side_opdict, _ = deca.decode(codedict)
            side_tex = side_opdict.get("uv_texture_gt")
            if side_tex is None:
                continue
            # Only fill pixels that front didn't cover
            side_valid = (side_tex.sum(dim=1, keepdim=True) > 0.1).float()
            fill_mask = side_valid * (1.0 - covered)
            base_tex = base_tex + side_tex * fill_mask
            covered = torch.clamp(covered + fill_mask, 0, 1)

        # 5. Apply blended texture
        front_opdict["uv_texture_gt"] = base_tex

        # 6. Export mesh with texture
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = str(Path(tmpdir) / "face.obj")
            deca.save_obj(obj_path, front_opdict)

            # Upscale texture to 1024px for higher detail
            from PIL import Image
            tex_path = str(Path(tmpdir) / "face.png")
            if Path(tex_path).exists():
                tex_img = Image.open(tex_path)
                tex_img = tex_img.resize((UV_SIZE, UV_SIZE), Image.LANCZOS)
                tex_img.save(tex_path)

            mesh = trimesh.load(
                obj_path,
                process=False,
                resolver=trimesh.visual.resolvers.FilePathResolver(tmpdir),
            )

            glb = _fix_glb_material(mesh.export(file_type="glb"))
            logger.info(f"GLB: {len(glb)} bytes, {len(mesh.vertices)} verts, "
                        f"{len(codedicts)} views, uv={UV_SIZE}")

    except Exception as e:
        logger.exception("Reconstruction error")
        raise HTTPException(500, str(e))

    return Response(content=glb, media_type="model/gltf-binary")
