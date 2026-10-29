"""
Crop Yield Prediction Endpoints
"""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
import numpy as np

from backend.app.schemas.models import PredictRequest, PredictResponse
from backend.app.data.storage import storage
from backend.app.models.inference import inference_service

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict_yield(req: PredictRequest):
    """
    Executes hyperlocal yield forecast for a selected parcel.
    Leverages genuine PyTorch ST-GNN, PureTemporalRNN, or baseline architecture
    with conformal prediction uncertainty calibration.
    """
    if storage.gdf is None or storage.timeseries_df is None:
        raise HTTPException(status_code=500, detail="Active dataset not loaded in storage.")
        
    if req.parcel_id not in list(storage.gdf["parcel_id"]):
        raise HTTPException(status_code=404, detail=f"Parcel '{req.parcel_id}' not found.")
        
    try:
        res = inference_service.predict_parcel(
            parcel_id=req.parcel_id,
            timeseries_df=storage.timeseries_df,
            gdf=storage.gdf,
            graph_data=storage.graph_data,
            labels_df=storage.labels_df,
            model_type=req.model_type,
            feature_set=req.feature_set,
            crop_type_override=req.crop_type
        )
        # Update cached prediction in storage
        storage.latest_predictions[req.parcel_id] = res["predicted_yield_tha"]
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/predictions")
def get_all_predictions():
    """Returns currently predicted yields across all parcels."""
    return storage.latest_predictions
