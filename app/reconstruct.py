import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from fastapi import HTTPException, UploadFile

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def run_vggt_on_dir(target_dir: Path, model: VGGT) -> dict[str, Any]:
    """
    Mirror the Gradio demo's run_model(...) helper.

    Expects images under target_dir / "images".
    Returns a predictions dict with numpy arrays and extra fields:
    - extrinsic, intrinsic, world_points_from_depth
    """
    images_dir = target_dir / "images"
    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file())

    if not image_paths:
        raise HTTPException(status_code=400, detail="No images found for reconstruction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else torch.no_grad():
            predictions: dict[str, Any] = model(images)

    # convert pose encoding to extrinsic/intrinsic
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic, intrinsic = extrinsic.squeeze(0), intrinsic.squeeze(0)
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # world points from depth
    depth_map = predictions["depth"].squeeze(0)  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    predictions["world_points_from_depth"] = world_points

    # move tensors to cpu + numpy (and remove batch dim)
    for key, value in list(predictions.items()):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions[key] = value

    torch.cuda.empty_cache()
    return predictions


def predictions_to_glb_minimal(predictions: dict[str, Any], confidence_threshhold: float) -> trimesh.Scene:
    """
    Minimal version of Meta's predictions_to_glb:
    - chooses world_points or world_points_from_depth
    - applies percentile confidence filtering
    - builds a point-cloud-only GLB
    """

    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # choose which 3D points to use
    if "world_points" in predictions:
        world_points = predictions["world_points"]
        conf = predictions.get("world_points_conf")
    else:
        world_points = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf")

    # fall back to uniform confidence if missing
    if conf is None:
        conf = np.ones(world_points.shape[:3], dtype=np.float32)

    images = predictions["images"]  # (S, H, W, 3) or similar
    # ensure NHWC
    if images.ndim == 4 and images.shape[1] == 3:
        # convert NCHW -> NHWC if needed
        images = np.transpose(images, (0, 2, 3, 1))

    vertices = world_points.reshape(-1, 3)
    colors = (images.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # percentile-based threshold, like the original helper
    confidence_threshhold = confidence_threshhold or 10.0

    if confidence_threshhold == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = float(np.percentile(conf_flat, confidence_threshhold))

    mask = (conf_flat >= conf_threshold) & (conf_flat > 1e-5)
    vertices = vertices[mask]
    colors = colors[mask]

    if vertices.size == 0:
        # degenerate scene, avoid crashing trimesh
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    return scene


def reconstruct_to_glb_from_uploads(
    files: list[UploadFile],
    model: VGGT,
    confidence_threshhold: float,
) -> Path:
    """
    Save uploaded images into a temp dir, run VGGT, build a GLB, and
    return the path to the GLB file.
    """
    if not isinstance(files, list):
        files = [files]
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    temp_root = Path(tempfile.mkdtemp(prefix="vggt_recon_"))
    images_dir = temp_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploads
        for idx, upload in enumerate(files):
            suffix = os.path.splitext(upload.filename or f"img_{idx}.png")[1] or ".png"
            out_path = images_dir / f"{idx:04d}{suffix}"
            with out_path.open("wb") as f:
                f.write(upload.file.read())

        predictions = run_vggt_on_dir(temp_root, model)
        scene = predictions_to_glb_minimal(predictions, confidence_threshhold=confidence_threshhold)

        glb_path = temp_root / "reconstruction.glb"
        scene.export(glb_path)

        return glb_path
    except Exception as exc:
        # cleanup and re-raise as HTTP error
        shutil.rmtree(temp_root, ignore_errors=True)
        import traceback
        traceback.print_exception(exc)
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {exc}") from exc
