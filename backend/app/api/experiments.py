"""
Experiments and Historical Evaluation Records
"""
from typing import Dict, Any, List
import os
import json
import glob
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Experiments"])


@router.get("/experiments")
def list_experiments():
    """Lists completed and saved model experiments with their performance metrics."""
    exp_files = glob.glob("experiments/*.json")
    results = []
    for f in exp_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if "experiment_id" in data:
                    results.append({
                        "experiment_id": data["experiment_id"],
                        "model_type": data.get("model_type"),
                        "feature_set": data.get("feature_set"),
                        "test_rmse": data.get("test_metrics", {}).get("rmse"),
                        "test_mae": data.get("test_metrics", {}).get("mae"),
                        "test_r2": data.get("test_metrics", {}).get("r2"),
                        "test_mape": data.get("test_metrics", {}).get("mape_percent"),
                        "duration_sec": data.get("duration_sec"),
                        "provenance": data.get("provenance", "synthetic_demonstration")
                    })
        except Exception:
            continue
            
    # Sort by test_rmse ascending
    results.sort(key=lambda x: x["test_rmse"] if x["test_rmse"] is not None else 999.0)
    return results


@router.get("/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    """Retrieves full configuration, loss history, and test metrics for an experiment."""
    fpath = os.path.join("experiments", f"{experiment_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
        
    with open(fpath, "r", encoding="utf-8") as fp:
        return json.load(fp)
