"""
Health and Status Endpoints
"""
from fastapi import APIRouter
import torch

from backend.app.schemas.models import HealthResponse
from backend.app.data.storage import storage

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health_status():
    return {
        "status": "healthy",
        "service": "CropYield-ST Hyperlocal Yield Forecasting API",
        "version": "1.0.0",
        "torch_available": True,
        "device": "cpu",
        "sample_data_loaded": storage.gdf is not None and len(storage.gdf) > 0
    }
