"""
CropYield-ST: Scientific Ablation Study and Cloud Robustness Experiment Module
Evaluates:
  1. Component Ablation (Optical only, SAR only, Fusion, Fusion+Temporal, Fusion+Temporal+Spatial GNN)
  2. Cloud Robustness (Impact of optical occlusion on yield forecasting accuracy)
"""
from typing import Dict, List, Optional, Any
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from backend.app.training.trainer import train_model_experiment


def run_full_ablation_study(
    timeseries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    gdf: Optional[Any] = None,
    epochs: int = 35,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the 5 key ablation experiments specified in the research report:
      1. Optical features only (ST-GNN on NDVI, EVI, NDRE, NDWI)
      2. SAR features only (ST-GNN on VV, VH, RVI, VV/VH ratio)
      3. Optical + SAR fusion (Baseline without graph: XGBoost / classical)
      4. Fusion + Temporal modeling (PureTemporalRNN without spatial graph)
      5. Fusion + Temporal + Spatial Graph modeling (Full Proposed ST-GNN)
    """
    ablation_configs = [
        {
            "name": "Mean-Yield Baseline",
            "model_type": "mean_baseline",
            "feature_set": "none",
            "description": "Predicts global training mean yield without satellite observations as naive baseline reference."
        },
        {
            "name": "Optical Features Only",
            "model_type": "st_gnn",
            "feature_set": "optical_only",
            "description": "Sentinel-2 MSI spectral indices (NDVI, EVI, NDRE, NDWI) with ST-GNN."
        },
        {
            "name": "SAR Features Only",
            "model_type": "st_gnn",
            "feature_set": "sar_only",
            "description": "Sentinel-1 C-band SAR backscatter (VV, VH, RVI, ratios) with ST-GNN."
        },
        {
            "name": "Optical + SAR Fusion (XGBoost)",
            "model_type": "xgboost",
            "feature_set": "fusion",
            "description": "Dual-sensor fusion using gradient boosted decision trees on seasonal aggregates."
        },
        {
            "name": "Fusion + Temporal Modeling (No Graph)",
            "model_type": "pure_temporal_rnn",
            "feature_set": "fusion",
            "description": "Dual-sensor fusion with temporal GRU encoder, without neighborhood graph convolution."
        },
        {
            "name": "CropYield-ST (Full ST-GNN)",
            "model_type": "st_gnn",
            "feature_set": "fusion",
            "description": "Proposed architecture: Dual-sensor fusion + Temporal GRU + Parcel Adjacency GCN."
        }
    ]
    
    results = []
    for cfg in ablation_configs:
        exp = train_model_experiment(
            timeseries_df=timeseries_df,
            labels_df=labels_df,
            gdf=gdf,
            model_type=cfg["model_type"],
            feature_set=cfg["feature_set"],
            epochs=epochs,
            random_seed=random_seed,
            experiment_id=f"ablation_{cfg['name'].lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')}"
        )
        results.append({
            "name": cfg["name"],
            "model_type": cfg["model_type"],
            "feature_set": cfg["feature_set"],
            "description": cfg["description"],
            "test_rmse": exp["test_metrics"]["rmse"],
            "test_mae": exp["test_metrics"]["mae"],
            "test_r2": exp["test_metrics"]["r2"],
            "test_mape": exp["test_metrics"]["mape_percent"],
            "duration_sec": exp["duration_sec"]
        })
        
    is_synth = True
    if "provenance" in labels_df.columns and len(labels_df["provenance"]) > 0:
        is_synth = "synthetic" in str(labels_df["provenance"].iloc[0])

    summary = {
        "study_name": "CropYield-ST Multi-Modal & Spatio-Temporal Ablation",
        "sample_count": len(labels_df),
        "provenance": "synthetic_demonstration" if is_synth else "real_satellite_data",
        "is_demonstration": is_synth,
        "evaluation_protocol": "Spatial block holdout partition; strictly non-overlapping train, calibration, and test fields.",
        "split": {
            "train_samples": exp["split"]["train_count"],
            "val_samples": exp["split"]["val_count"],
            "test_samples": exp["split"]["test_count"]
        },
        "results": results
    }
    
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/ablation_study_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary


def run_cloud_robustness_study(
    timeseries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    gdf: Optional[Any] = None,
    epochs: int = 30,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates model resilience under varying levels of simulated cloud corruption
    (0%, 25%, 50%, 75% optical degradation) comparing Optical-only vs ST-GNN Fusion.
    Empirically illustrates why C-band SAR backscatter prevents yield error explosions!
    """
    cloud_levels = [0.0, 0.25, 0.50, 0.75]
    records = []
    
    np.random.seed(random_seed)
    
    for level in cloud_levels:
        corrupted_ts = timeseries_df.copy()
        if level > 0:
            # Mask random optical rows as cloudy
            mask = np.random.rand(len(corrupted_ts)) < level
            corrupted_ts.loc[mask, "ndvi"] = 0.15 + np.random.normal(0, 0.05, sum(mask))
            corrupted_ts.loc[mask, "optical_valid"] = False
            
        # Train Optical-only ST-GNN
        opt_exp = train_model_experiment(
            timeseries_df=corrupted_ts,
            labels_df=labels_df,
            gdf=gdf,
            model_type="st_gnn",
            feature_set="optical_only",
            epochs=epochs,
            random_seed=random_seed
        )
        
        # Train Full Optical+SAR ST-GNN
        fusion_exp = train_model_experiment(
            timeseries_df=corrupted_ts,
            labels_df=labels_df,
            gdf=gdf,
            model_type="st_gnn",
            feature_set="fusion",
            epochs=epochs,
            random_seed=random_seed
        )
        
        records.append({
            "cloud_occlusion_pct": int(level * 100),
            "optical_only_rmse": opt_exp["test_metrics"]["rmse"],
            "optical_only_r2": opt_exp["test_metrics"]["r2"],
            "fusion_st_gnn_rmse": fusion_exp["test_metrics"]["rmse"],
            "fusion_st_gnn_r2": fusion_exp["test_metrics"]["r2"]
        })
        
    summary = {
        "experiment_name": "Cloud Occlusion Robustness Analysis",
        "description": "Compares performance degradation under severe cloud cover scenarios",
        "data": records
    }
    
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/cloud_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary
