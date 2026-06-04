from __future__ import annotations
import io
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from mc_dropout.predict import mc_predict

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}


class PredictionResponse(BaseModel):
    prediction: str
    mean_probability: float
    uncertainty: float
    histogram_b64: str
    contour_b64: Optional[str] = None
    scatter_b64: Optional[str] = None
    area_px: Optional[float] = None
    mc_area_samples: Optional[int] = None


class BatchPredictionItem(BaseModel):
    filename: str
    prediction: Optional[str] = None
    mean_probability: Optional[float] = None
    uncertainty: Optional[float] = None
    histogram_b64: Optional[str] = None
    contour_b64: Optional[str] = None
    scatter_b64: Optional[str] = None
    area_px: Optional[float] = None
    mc_area_samples: Optional[int] = None
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    name: str
    path: str
    size_mb: float
    loaded_at: str


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    info = getattr(request.app.state, "model_info", None)
    if info is None:
        raise HTTPException(status_code=503, detail="Model info not available.")
    return ModelInfoResponse(**info)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — server is still starting up.")

    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Could not decode the uploaded file as an image.")

    config = request.app.state.config
    device = request.app.state.device

    try:
        result = mc_predict(
            image=image,
            model=model,
            num_samples=config.inference.num_mc_samples,
            threshold=config.inference.threshold,
            image_size=config.dataset.image_size,
            device=device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=List[BatchPredictionItem])
async def predict_batch(
    request: Request, files: List[UploadFile] = File(...)
) -> List[BatchPredictionItem]:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — server is still starting up.")

    config = request.app.state.config
    device = request.app.state.device
    results: List[BatchPredictionItem] = []

    for file in files:
        fname = file.filename or "unknown"

        if file.content_type not in _ALLOWED_CONTENT_TYPES:
            results.append(BatchPredictionItem(
                filename=fname,
                error=f"Unsupported file type '{file.content_type}'.",
            ))
            continue

        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            results.append(BatchPredictionItem(filename=fname, error="Could not decode as an image."))
            continue

        try:
            result = mc_predict(
                image=image,
                model=model,
                num_samples=config.inference.num_mc_samples,
                threshold=config.inference.threshold,
                image_size=config.dataset.image_size,
                device=device,
            )
            results.append(BatchPredictionItem(filename=fname, **result))
        except Exception as exc:
            results.append(BatchPredictionItem(filename=fname, error=f"Inference failed: {exc}"))

    return results
