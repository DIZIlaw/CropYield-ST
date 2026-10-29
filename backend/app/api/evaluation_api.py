"""
Model Comparison, Ablation Studies, and Cloud Robustness Analysis Endpoints
"""
from typing import Dict, Any, List
import os
import json
from fastapi import APIRouter, HTTPException

from backend.app.data.storage import storage
from backend.app.evaluation.ablation import run_full_ablation_study, run_cloud_robustness_study

router = APIRouter(tags=["Evaluation"])


@router.get("/evaluation/ablation")
def get_ablation_results():
    """Returns saved or computed ablation study comparison across all 5 configurations."""
    cache_path = "experiments/ablation_study_summary.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # If not cached, execute now
    if storage.timeseries_df is None or storage.labels_df is None:
        raise HTTPException(status_code=400, detail="Datasets not loaded.")
        
    return run_full_ablation_study(
        timeseries_df=storage.timeseries_df,
        labels_df=storage.labels_df,
        gdf=storage.gdf,
        epochs=25
    )


@router.post("/evaluation/ablation/run")
def trigger_ablation_study():
    """Forces re-execution of the full ablation study."""
    if storage.timeseries_df is None or storage.labels_df is None:
        raise HTTPException(status_code=400, detail="Datasets not loaded.")
        
    return run_full_ablation_study(
        timeseries_df=storage.timeseries_df,
        labels_df=storage.labels_df,
        gdf=storage.gdf,
        epochs=30
    )


@router.get("/evaluation/cloud-robustness")
def get_cloud_robustness_results():
    """Returns cloud occlusion resilience metrics comparing Optical-only vs ST-GNN."""
    cache_path = "experiments/cloud_robustness_summary.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    if storage.timeseries_df is None or storage.labels_df is None:
        raise HTTPException(status_code=400, detail="Datasets not loaded.")
        
    return run_cloud_robustness_study(
        timeseries_df=storage.timeseries_df,
        labels_df=storage.labels_df,
        gdf=storage.gdf,
        epochs=20
    )
