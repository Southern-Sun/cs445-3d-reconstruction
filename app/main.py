from contextlib import asynccontextmanager
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


model: VGGT | None = None
device: str = "cpu"
dtype: torch.dtype = torch.float32


class PredictionSummary(BaseModel):
    num_images: int
    has_cameras: bool
    has_depths: bool
    has_point_maps: bool


def init_model() -> None:
    """
    Initialize VGGT and global device/dtype.
    Called once at startup.
    """
    global model, device, dtype

    if torch.cuda.is_available():
        device = "cuda"
        major, _ = torch.cuda.get_device_capability()
        # per VGGT docs: bfloat16 for Ampere+ (cc >= 8.0), else float16
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    # This will download weights from HF the first time it runs.
    # You can swap to model = VGGT() + manual load_state_dict if you prefer.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, Any]:
    # On startup
    init_model()
    yield
    # On shutdown

app = FastAPI(title="VGGT API", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok"}


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
