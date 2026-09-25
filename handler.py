"""RunPod Serverless handler for Hunyuan3D-2.1 face reconstruction.

Receives base64 face photos, generates a textured 3D mesh,
and returns the GLB as base64.
"""

import base64
import logging
import tempfile
import traceback
from pathlib import Path

import runpod

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("=== Handler module starting ===")
logger.info(f"Python: {__import__('sys').version}")

# Test basic imports at startup so errors show in logs
try:
    import torch
    logger.info(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except Exception as e:
    logger.error(f"PyTorch import failed: {e}", exc_info=True)

try:
    import hy3dgen
    logger.info(f"hy3dgen imported OK from {hy3dgen.__file__}")
except Exception as e:
    logger.error(f"hy3dgen import failed: {e}", exc_info=True)

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    logger.info("shapegen import OK")
except Exception as e:
    logger.error(f"shapegen import failed: {e}", exc_info=True)

try:
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    logger.info("texgen import OK")
except Exception as e:
    logger.error(f"texgen import failed: {e}", exc_info=True)

logger.info("=== Imports done, starting RunPod handler ===")

# Lazy-loaded pipelines (loaded on first request, not at startup)
_shape_pipeline = None
_tex_pipeline = None


def _load_pipelines():
    global _shape_pipeline, _tex_pipeline

    if _shape_pipeline is not None:
        return

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    logger.info("Loading shape pipeline...")
    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    _shape_pipeline.to("cuda")
    logger.info("Shape pipeline loaded.")

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        logger.info("Loading texture pipeline...")
        _tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-paint-v2-0",
        )
        _tex_pipeline.to("cuda")
        logger.info("Texture pipeline loaded.")
    except Exception as e:
        logger.warning(f"Texture pipeline not available: {e}")
        _tex_pipeline = None


def handler(job):
    """RunPod handler: receives views, returns GLB base64."""
    try:
        job_input = job["input"]
        views = job_input.get("views", [])

        if not views:
            return {"error": "No views provided"}

        _load_pipelines()

        with tempfile.TemporaryDirectory(prefix="hunyuan_") as tmp_dir:
            tmp = Path(tmp_dir)

            # Save input images to temp files
            photos = {}
            for view in views:
                pose = view.get("pose", "")
                b64 = view.get("image_base64", "")
                if pose == "eyes" or not b64:
                    continue
                p = tmp / f"{pose}.jpg"
                p.write_bytes(base64.b64decode(b64))
                photos[pose] = p

            if "front" not in photos:
                return {"error": "Missing front view"}

            from PIL import Image

            front_img = Image.open(str(photos["front"])).convert("RGB")

            # Step 1: Generate shape
            logger.info("Generating shape...")
            mesh = _shape_pipeline(
                image=front_img,
                num_inference_steps=30,
                guidance_scale=5.0,
                octree_resolution=256,
                num_chunks=8000,
            )[0]

            # Step 2: Generate texture (if available)
            if _tex_pipeline is not None:
                logger.info("Generating texture...")
                mesh = _tex_pipeline(mesh, image=front_img)
            else:
                logger.warning("Skipping texture — pipeline not loaded")

            # Step 3: Export as GLB
            glb_path = tmp / "output.glb"
            mesh.export(str(glb_path))
            glb_bytes = glb_path.read_bytes()

            logger.info(f"GLB generated: {len(glb_bytes)} bytes")

            return {
                "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
                "size_bytes": len(glb_bytes),
            }

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
