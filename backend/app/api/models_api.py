"""
Model Training and Model Catalog Endpoints
"""
from typing import Dict, Any, List
import uuid
import threading
from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.app.schemas.models import TrainRequest
from backend.app.data.storage import storage
from backend.app.training.trainer import train_model_experiment

router = APIRouter(tags=["Models"])

# In-memory training jobs tracker
training_jobs: Dict[str, Dict[str, Any]] = {}


def _run_training_job(job_id: str, req: TrainRequest):
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 10
        
        res = train_model_experiment(
            timeseries_df=storage.timeseries_df,
            labels_df=storage.labels_df,
            gdf=storage.gdf,
            model_type=req.model_type,
            feature_set=req.feature_set,
            epochs=req.epochs,
            lr=req.learning_rate,
            hidden_dim=req.hidden_dim,
            dropout=req.dropout,
            random_seed=req.random_seed,
            experiment_id=job_id
        )
        
        # Update predictions in storage with new model predictions
        if "predictions" in res:
            storage.latest_predictions.update(res["predictions"])
            
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["result"] = res
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


@router.post("/models/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Initiates a model training experiment."""
    if storage.labels_df is None:
        raise HTTPException(status_code=400, detail="Cannot train model without yield ground-truth labels.")
        
    job_id = f"job_{req.model_type}_{uuid.uuid4().hex[:8]}"
    training_jobs[job_id] = {
        "job_id": job_id,
        "model_type": req.model_type,
        "feature_set": req.feature_set,
        "epochs": req.epochs,
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None
    }
    
    # Run in background
    background_tasks.add_task(_run_training_job, job_id, req)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Training job '{job_id}' started in background."
    }


@router.get("/models/train/{job_id}")
def get_training_status(job_id: str):
    """Queries real-time progress and results of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return training_jobs[job_id]


@router.get("/models")
def list_available_models():
    """Lists eligible models and architectures available in CropYield-ST."""
    return [
        {
            "id": "st_gnn",
            "name": "CropYield-ST (Spatio-Temporal Graph Neural Network)",
            "category": "Proposed Architecture",
            "components": ["Sentinel-1/2 Input Fusion", "GRU/LSTM Temporal Trajectory Encoder", "Parcel Adjacency GCN", "MLP Yield Regressor"],
            "description": "Combines temporal crop growth dynamics with spatial neighbor contiguity."
        },
        {
            "id": "pure_temporal_rnn",
            "name": "Temporal Deep Learning (No Graph)",
            "category": "Temporal Baseline",
            "components": ["Sentinel-1/2 Input Fusion", "GRU/LSTM Temporal Trajectory Encoder", "MLP Yield Regressor"],
            "description": "Models multi-sensor phenology curves without spatial parcel graph interactions."
        },
        {
            "id": "xgboost",
            "name": "Gradient Boosted Trees (XGBoost)",
            "category": "Classical ML Baseline",
            "components": ["Seasonal Aggregate Statistics (Mean, Max, Min, Std)"],
            "description": "Standard benchmark tree ensemble on tabular summary features."
        },
        {
            "id": "random_forest",
            "name": "Random Forest Regressor",
            "category": "Classical ML Baseline",
            "components": ["Seasonal Aggregate Statistics"],
            "description": "Bagging ensemble of decision trees on multi-sensor indices."
        }
    ]
