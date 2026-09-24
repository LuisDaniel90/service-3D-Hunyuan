"""RunPod Serverless handler for Hunyuan3D-2.1 face reconstruction.

Receives base64 face photos, generates a textured 3D mesh,
and returns the GLB as base64.
"""

import base64
import logging
import tempfile
from pathlib import Path

import runpod
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model at startup (warm worker) ─────────────────────────────
# Model is loaded once when the container starts. Subsequent requests
# reuse the loaded pipeline without cold-start overhead.

logger.info("Loading Hunyuan3D-2.1 pipeline...")

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2",
    subfolder="hunyuan3d-dit-v2-0",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
shape_pipeline.to("cuda")

tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    "tencent/Hunyuan3D-2",
    subfolder="hunyuan3d-paint-v2-0",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
tex_pipeline.to("cuda")

logger.info("Pipelines loaded.")


def _save_image(b64: str, path: Path) -> Path:
    """Decode a base64 image and save to disk."""
    path.write_bytes(base64.b64decode(b64))
    return path


def handler(job):
    """RunPod handler: receives views, returns GLB base64."""
    job_input = job["input"]
    views = job_input.get("views", [])

    if not views:
        return {"error": "No views provided"}

    with tempfile.TemporaryDirectory(prefix="hunyuan_") as tmp_dir:
        tmp = Path(tmp_dir)

        # Save input images to temp files
        photos = {}
        for view in views:
            pose = view.get("pose", "")
            b64 = view.get("image_base64", "")
            if pose == "eyes" or not b64:
                continue
            photos[pose] = _save_image(b64, tmp / f"{pose}.jpg")

        if "front" not in photos:
            return {"error": "Missing front view"}

        # Use front as primary input
        from PIL import Image

        front_img = Image.open(str(photos["front"])).convert("RGB")

        # Collect multi-view images if available
        mv_images = {}
        if "left" in photos or "left45" in photos:
            p = photos.get("left") or photos.get("left45")
            mv_images["left"] = Image.open(str(p)).convert("RGB")
        if "right" in photos or "right45" in photos:
            p = photos.get("right") or photos.get("right45")
            mv_images["right"] = Image.open(str(p)).convert("RGB")

        # Step 1: Generate shape
        logger.info("Generating shape...")
        mesh = shape_pipeline(
            image=front_img,
            num_inference_steps=30,
            guidance_scale=5.0,
            octree_resolution=256,
            num_chunks=8000,
        )[0]

        # Step 2: Generate texture
        logger.info("Generating texture...")
        textured_mesh = tex_pipeline(mesh, image=front_img)

        # Step 3: Export as GLB
        glb_path = tmp / "output.glb"
        textured_mesh.export(str(glb_path))
        glb_bytes = glb_path.read_bytes()

        logger.info(f"GLB generated: {len(glb_bytes)} bytes")

        return {
            "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
            "size_bytes": len(glb_bytes),
        }


runpod.serverless.start({"handler": handler})
