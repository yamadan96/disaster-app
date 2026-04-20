"""FastAPI application for disaster building damage assessment."""

import io
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.predictor import CLASS_NAMES, Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_CLASSES = 6


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    rejected: bool


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the predictor on startup, clean up on shutdown."""
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir is None:
        raise RuntimeError(
            "CHECKPOINT_DIR environment variable is required. "
            "Set it to the directory containing best_model.pth."
        )

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise RuntimeError(f"CHECKPOINT_DIR does not exist: {checkpoint_path}")

    device = os.environ.get("DEVICE", "cuda")
    predictor = Predictor()
    predictor.initialize(checkpoint_dir=checkpoint_path, device=device)
    logger.info("FastAPI application started with model on %s", device)

    yield

    logger.info("FastAPI application shutting down.")


app = FastAPI(
    title="Disaster Building Damage Assessment API",
    description="Classify building damage from disaster images using DINOv2 + LoRA.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """Predict building damage class from an uploaded image.

    Parameters
    ----------
    file : UploadFile
        Image file (JPEG, PNG, etc.).

    Returns
    -------
    PredictionResponse
        Prediction result with class, confidence, and rejection flag.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. "
            "Please upload an image file.",
        )

    try:
        contents = await file.read()
    except Exception:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(
            status_code=400,
            detail="Failed to read uploaded file.",
        ) from None

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        logger.exception("Failed to decode image")
        raise HTTPException(
            status_code=400,
            detail="Failed to decode the uploaded file as an image.",
        ) from None

    predictor = Predictor()

    try:
        result = predictor.predict(image)
    except RuntimeError:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Please try again later.",
        ) from None

    probabilities = {
        CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(result.probabilities)
    }

    return PredictionResponse(
        class_id=result.class_id,
        class_name=result.class_name,
        confidence=round(result.confidence, 4),
        probabilities=probabilities,
        rejected=result.rejected,
    )
