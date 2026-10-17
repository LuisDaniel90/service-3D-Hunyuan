"""RunPod Serverless handler — 3D face photogrammetry with COLMAP.

Receives a video (base64) OR multiple photos from different angles,
extracts frames, runs Structure-from-Motion + Multi-View Stereo,
and returns a textured GLB mesh faithful to the real patient.
"""

import base64
import logging
import tempfile
import traceback
from pathlib import Path

import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("=== COLMAP Photogrammetry Handler starting ===")
logger.info(f"Python: {__import__('sys').version}")

try:
    import pycolmap
    logger.info(f"pycolmap {pycolmap.__version__}")
except Exception as e:
    logger.error(f"pycolmap import failed: {e}", exc_info=True)

try:
    import cv2
    logger.info(f"OpenCV {cv2.__version__}")
except Exception as e:
    logger.error(f"OpenCV import failed: {e}", exc_info=True)

try:
    import numpy as np  # noqa: F811
    import trimesh
    import open3d as o3d
    logger.info("numpy, trimesh, open3d OK")
except Exception as e:
    logger.error(f"Import failed: {e}", exc_info=True)

logger.info("=== Imports done ===")

# ── Frame extraction from video ──────────────────────────────────────

def extract_frames_from_video(video_path: Path, output_dir: Path, max_frames: int = 40) -> int:
    """Extract evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video frames")

    step = max(1, total_frames // max_frames)
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0 and count < max_frames:
            out_path = output_dir / f"frame_{count:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {count} frames from {total_frames} total (step={step})")
    return count


# ── COLMAP reconstruction pipeline ───────────────────────────────────

def run_colmap_reconstruction(images_dir: Path, workspace: Path) -> Path:
    """Run COLMAP SfM + MVS pipeline. Returns path to fused PLY."""
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    logger.info("COLMAP: extracting features...")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
    )

    logger.info("COLMAP: matching features (exhaustive)...")
    pycolmap.match_exhaustive(database_path=db_path)

    logger.info("COLMAP: incremental mapping (SfM)...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=images_dir,
        output_path=sparse_dir,
    )

    if not reconstructions:
        raise RuntimeError("COLMAP SfM failed — could not reconstruct from images. "
                           "Ensure the face is visible from multiple angles.")

    best = max(reconstructions, key=lambda r: reconstructions[r].num_reg_images())
    logger.info(f"SfM: {reconstructions[best].num_reg_images()} registered images, "
                f"{reconstructions[best].num_points3D()} points")

    # Write the best reconstruction for dense step
    best_sparse = sparse_dir / "0"
    best_sparse.mkdir(exist_ok=True)
    reconstructions[best].write(best_sparse)

    logger.info("COLMAP: undistorting images...")
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=best_sparse,
        image_path=images_dir,
    )

    logger.info("COLMAP: patch match stereo (dense)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir)

    logger.info("COLMAP: stereo fusion...")
    fused_path = dense_dir / "fused.ply"
    pycolmap.stereo_fusion(
        output_path=fused_path,
        workspace_path=dense_dir,
    )

    if not fused_path.exists() or fused_path.stat().st_size < 1000:
        raise RuntimeError("Dense fusion produced empty result")

    logger.info(f"Fused point cloud: {fused_path.stat().st_size / 1e6:.1f} MB")
    return fused_path


# ── Point cloud → textured mesh → GLB ────────────────────────────────

def pointcloud_to_glb(ply_path: Path, output_path: Path) -> None:
    """Convert a colored point cloud to a textured GLB mesh."""
    logger.info("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_points = len(pcd.points)
    logger.info(f"Point cloud: {n_points} points")

    if n_points < 100:
        raise RuntimeError(f"Too few points ({n_points}) for mesh reconstruction")

    # Statistical outlier removal
    logger.info("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals
    logger.info("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson surface reconstruction
    logger.info("Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )

    # Remove low-density vertices (noisy edges)
    import numpy as np
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()

    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # Convert to trimesh for GLB export (Open3D doesn't export GLB with vertex colors well)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if colors is not None and len(colors) == len(vertices):
        tmesh.visual.vertex_colors = (colors * 255).astype(np.uint8)

    glb_bytes = tmesh.export(file_type="glb")
    output_path.write_bytes(glb_bytes)
    logger.info(f"GLB exported: {len(glb_bytes)} bytes")


# ── RunPod handler ────────────────────────────────────────────────────

def handler(job):
    """Receives video or photos, returns textured 3D GLB."""
    try:
        job_input = job["input"]

        with tempfile.TemporaryDirectory(prefix="colmap_") as tmp_dir:
            tmp = Path(tmp_dir)
            images_dir = tmp / "images"
            images_dir.mkdir()
            workspace = tmp / "workspace"
            workspace.mkdir()

            frame_count = 0

            # Option 1: Video input (preferred — more frames, better reconstruction)
            video_b64 = job_input.get("video_base64")
            if video_b64:
                video_ext = job_input.get("video_ext", "mp4")
                video_path = tmp / f"input.{video_ext}"
                video_path.write_bytes(base64.b64decode(video_b64))
                max_frames = job_input.get("max_frames", 40)
                frame_count = extract_frames_from_video(video_path, images_dir, max_frames)

            # Option 2: Multiple photos
            views = job_input.get("views", [])
            if views and frame_count == 0:
                for i, view in enumerate(views):
                    b64 = view.get("image_base64", "")
                    if not b64:
                        continue
                    pose = view.get("pose", f"view_{i}")
                    out_path = images_dir / f"{pose}.jpg"
                    img_bytes = base64.b64decode(b64)
                    out_path.write_bytes(img_bytes)
                    frame_count += 1

            if frame_count < 3:
                return {"error": f"Need at least 3 images for reconstruction, got {frame_count}"}

            logger.info(f"Starting reconstruction with {frame_count} images")

            # Run COLMAP
            fused_ply = run_colmap_reconstruction(images_dir, workspace)

            # Convert to GLB
            glb_path = tmp / "output.glb"
            pointcloud_to_glb(fused_ply, glb_path)

            glb_bytes = glb_path.read_bytes()

            return {
                "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
                "size_bytes": len(glb_bytes),
                "num_frames": frame_count,
                "method": "colmap_photogrammetry",
            }

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
