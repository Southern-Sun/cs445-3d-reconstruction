from contextlib import asynccontextmanager
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator
import threading

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from .reconstruct import reconstruct_to_glb_from_uploads, reconstruct_from_video

_model_lock = threading.Lock()
model: VGGT | None = None
device: str = "cpu"
dtype: torch.dtype = torch.float32

def get_model() -> VGGT:
    """
    Lazily initialize VGGT on first use.
    Ensures only one thread does the heavy load.
    """
    global model, device, dtype

    if model is not None:
        return model

    with _model_lock:
        # Double-check inside lock
        if model is not None:
            return model

        if torch.cuda.is_available():
            device = "cuda"
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        loaded = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        loaded.eval()
        model = loaded
        print("VGGT model loaded on", device)

        return model


class PredictionSummary(BaseModel):
    num_images: int
    has_cameras: bool
    has_depths: bool
    has_point_maps: bool


app = FastAPI(title="VGGT API", version="0.2.0")

@app.get("/health")
def health() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok"}

@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionSummary)
async def predict(files: list[UploadFile] = File(...)) -> PredictionSummary:
    """
    Basic demo endpoint:
    - Accepts multiple image files.
    - Runs VGGT once on all of them.
    - Returns a lightweight summary (no giant tensors in JSON).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Save uploads to a temp directory because VGGT's loader expects paths
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        image_paths: list[str] = []
        for f in files:
            target = tmp_path / f.filename
            # Read whole file into memory, then write
            contents = await f.read()
            target.write_bytes(contents)
            image_paths.append(str(target))

        if not image_paths:
            raise HTTPException(status_code=400, detail="No images provided")

        # Load + preprocess via VGGT utilities
        images = load_and_preprocess_images(image_paths).to(device)

        assert model is not None  # for type checkers
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions: dict[str, Any] = model(images)
            else:
                predictions = model(images)

    # Predictions is a dict-like with keys like "cameras", "depth_maps", etc.
    has_cameras = "cameras" in predictions
    has_depths = "depth_maps" in predictions
    has_point_maps = "point_maps" in predictions

    return PredictionSummary(
        num_images=images.shape[1],
        has_cameras=bool(has_cameras),
        has_depths=bool(has_depths),
        has_point_maps=bool(has_point_maps),
    )

@app.post("/reconstruct")
async def reconstruct_endpoint(
    files: list[UploadFile] = File(...),
    confidence_threshhold: float = Query(50.0, description="Percentile of low-confidence points to drop"),
) -> FileResponse:
    """
    Upload images + confidence threshold, get a GLB point cloud back.
    """
    vggt_model = get_model()

    glb_path = reconstruct_to_glb_from_uploads(files=files, model=vggt_model, confidence_threshhold=confidence_threshhold)

    # Let the OS clean temp dirs after process exit; if you want more aggressive cleanup,
    # you can schedule a background task to delete glb_path.parent
    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename="reconstruction.glb",
    )

@app.post("/reconstruct_video")
async def reconstruct_video_endpoint(
    video: UploadFile = File(...),
    conf_thres: float = Query(50.0),
    max_frames: int = Query(200),
    sample_rate: int = Query(1, description="Use every Nth frame"),
):
    """
    Accept a single video upload and return a GLB point cloud reconstruction.
    """
    vggt_model = get_model()

    glb_path = reconstruct_from_video(
        video=video,
        model=vggt_model,
        conf_thres=conf_thres,
        max_frames=max_frames,
        sample_rate=sample_rate,
    )

    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename="reconstruction.glb",
    )
