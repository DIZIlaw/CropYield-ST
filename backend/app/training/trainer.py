"""
CropYield-ST: Model Training and Experiment Execution Engine
Trains ST-GNN and baseline models with spatial separation, validation tracking, and checkpoint persistence.
"""
from typing import Dict, List, Optional, Tuple, Any
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from backend.app.models.st_gnn import CropYieldSTGNN
from backend.app.models.baselines import PureTemporalRNN, ClassicalMLBaseline
from backend.app.features.temporal_features import (
    TemporalFeatureProcessor,
    extract_tabular_summary_features,
    DEFAULT_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    SAR_FEATURE_COLUMNS
)
from backend.app.geospatial.parcel_graph import build_parcel_graph_from_geodataframe
from backend.app.training.spatial_split import spatial_block_split
from backend.app.evaluation.metrics import compute_regression_metrics


def train_model_experiment(
    timeseries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    gdf: Optional[Any] = None,
    model_type: str = "st_gnn",
    feature_set: str = "fusion", # "fusion", "optical_only", "sar_only"
    epochs: int = 50,
    lr: float = 0.005,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    random_seed: int = 42,
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a complete reproducible training and evaluation experiment.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    start_time = time.time()
    exp_id = experiment_id or f"exp_{model_type}_{feature_set}_{int(start_time)}"
    
    # Select feature columns based on ablation mode
    if feature_set == "optical_only":
        feature_cols = OPTICAL_FEATURE_COLUMNS
    elif feature_set == "sar_only":
        feature_cols = SAR_FEATURE_COLUMNS
    else:
        feature_cols = DEFAULT_FEATURE_COLUMNS
        
    # Align parcel IDs
    common_pids = sorted(list(set(timeseries_df["parcel_id"]).intersection(set(labels_df["parcel_id"]))))
    if len(common_pids) < 4:
        raise ValueError(f"Insufficient labeled parcels ({len(common_pids)}) for training. Minimum 4 required.")
        
    # Spatial split
    split = spatial_block_split(common_pids, gdf=gdf, test_ratio=0.25, val_ratio=0.15, random_seed=random_seed)
    train_pids = split["train"]
    val_pids = split["val"]
    test_pids = split["test"]
    
    # Build parcel graph if gdf available
    if gdf is not None:
        gdf_filtered = gdf[gdf["parcel_id"].isin(common_pids)].copy()
        graph_data = build_parcel_graph_from_geodataframe(gdf_filtered, id_col="parcel_id")
        adj_norm = torch.tensor(graph_data["normalized_adjacency"], dtype=torch.float32)
    else:
        # Fallback identity graph
        n = len(common_pids)
        adj_norm = torch.eye(n, dtype=torch.float32)
        
    # Process temporal features
    processor = TemporalFeatureProcessor(feature_cols=feature_cols, impute_cloudy=True)
    processor.fit(timeseries_df, train_parcel_ids=train_pids)
    X_tensor, _ = processor.transform(timeseries_df, ordered_parcel_ids=common_pids)
    
    # Ground truth yield targets
    labels_indexed = labels_df.set_index("parcel_id")
    y_vals = np.array([float(labels_indexed.loc[pid, "yield_tha"]) for pid in common_pids], dtype=np.float32)
    
    # Target normalization (fit on train only)
    pid_to_idx = {pid: i for i, pid in enumerate(common_pids)}
    train_idx = [pid_to_idx[p] for p in train_pids]
    val_idx = [pid_to_idx[p] for p in val_pids]
    test_idx = [pid_to_idx[p] for p in test_pids]
    
    y_mean = float(np.mean(y_vals[train_idx]))
    y_std = float(np.std(y_vals[train_idx])) + 1e-6
    y_norm = (y_vals - y_mean) / y_std
    
    # Device
    device = torch.device("cpu")
    
    # Branch 1: Mean-Yield Baseline
    if model_type == "mean_baseline":
        y_pred_all = np.full(len(common_pids), y_mean, dtype=np.float32)
        test_metrics = compute_regression_metrics(y_vals[test_idx], y_pred_all[test_idx])
        val_metrics = compute_regression_metrics(y_vals[val_idx], y_pred_all[val_idx])
        train_metrics = compute_regression_metrics(y_vals[train_idx], y_pred_all[train_idx])
        loss_history = []
        model = None
    # Branch 2: Classical ML Baseline (XGBoost / Random Forest)
    elif model_type in ("xgboost", "random_forest"):
        X_tab = extract_tabular_summary_features(timeseries_df, ordered_parcel_ids=common_pids).values
        clf = ClassicalMLBaseline(model_type=model_type, random_state=random_seed)
        clf.fit(X_tab[train_idx], y_vals[train_idx])
        
        y_pred_all = clf.predict(X_tab)
        test_metrics = compute_regression_metrics(y_vals[test_idx], y_pred_all[test_idx])
        val_metrics = compute_regression_metrics(y_vals[val_idx], y_pred_all[val_idx])
        train_metrics = compute_regression_metrics(y_vals[train_idx], y_pred_all[train_idx])
        
        loss_history = []
        model = clf
    else:
        # Branch 3: Deep Learning models (ST-GNN or PureTemporalRNN)
        X_pt = torch.tensor(X_tensor, dtype=torch.float32, device=device)
        y_pt = torch.tensor(y_norm, dtype=torch.float32, device=device).unsqueeze(-1)
        adj_pt = adj_norm.to(device)
        
        in_dim = len(feature_cols)
        h_dim = min(hidden_dim, 32)
        drop = min(dropout, 0.08)
        if model_type == "st_gnn":
            model = CropYieldSTGNN(
                in_features=in_dim,
                temporal_dim=h_dim,
                spatial_dim=h_dim,
                num_rnn_layers=1,
                num_gcn_layers=1,
                dropout=drop
            ).to(device)
        else: # pure_temporal_rnn
            model = PureTemporalRNN(
                in_features=in_dim,
                hidden_dim=h_dim,
                num_layers=1,
                dropout=drop
            ).to(device)
            
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        
        loss_history = []
        best_val_loss = float("inf")
        best_state = None
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            if model_type == "st_gnn":
                preds, _, _ = model(X_pt, adj_pt)
            else:
                preds, _, _ = model(X_pt)
                
            loss = criterion(preds[train_idx], y_pt[train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if model_type == "st_gnn":
                    v_preds, _, _ = model(X_pt, adj_pt)
                else:
                    v_preds, _, _ = model(X_pt)
                val_loss = float(criterion(v_preds[val_idx], y_pt[val_idx]).item())
                
            scheduler.step(val_loss)
            
            # Checkpoint selection: only consider checkpoints once model has begun
            # learning training distribution (epoch >= 10, train_loss < 0.70)
            # to avoid reverting to untrained random initialization predicting the mean
            if epoch >= 10 and loss.item() < 0.70 and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
            loss_history.append({
                "epoch": epoch,
                "train_loss": round(float(loss.item()), 5),
                "val_loss": round(val_loss, 5)
            })
            
        if best_state is not None:
            model.load_state_dict(best_state)
            
        # Final Evaluation on test set
        model.eval()
        with torch.no_grad():
            if model_type == "st_gnn":
                all_preds_norm, _, _ = model(X_pt, adj_pt)
            else:
                all_preds_norm, _, _ = model(X_pt)
                
        all_preds_norm = all_preds_norm.squeeze(-1).cpu().numpy()
        # Inverse transform
        y_pred_all = (all_preds_norm * y_std) + y_mean
        
        test_metrics = compute_regression_metrics(y_vals[test_idx], y_pred_all[test_idx])
        val_metrics = compute_regression_metrics(y_vals[val_idx], y_pred_all[val_idx])
        train_metrics = compute_regression_metrics(y_vals[train_idx], y_pred_all[train_idx])
        
    duration_sec = round(time.time() - start_time, 2)
    
    # 1. Calibration on held-out validation residuals
    val_residuals = [float(r) for r in np.abs(y_vals[val_idx] - y_pred_all[val_idx])]
    m_cal = len(val_residuals)
    if m_cal >= 3:
        q_level = min(1.0, np.ceil((m_cal + 1) * 0.95) / m_cal)
        conformal_q95 = float(np.quantile(val_residuals, min(1.0, q_level)))
    else:
        conformal_q95 = 0.50

    # 2. Evaluation on INDEPENDENT held-out test set
    test_y_true = y_vals[test_idx]
    test_y_pred = y_pred_all[test_idx]
    test_lower = np.maximum(0.0, test_y_pred - conformal_q95)
    test_upper = test_y_pred + conformal_q95
    in_interval = (test_y_true >= test_lower) & (test_y_true <= test_upper)
    empirical_test_coverage = float(np.mean(in_interval) * 100.0)
    mean_test_width = float(np.mean(test_upper - test_lower))
    
    # Statistical validation requires adequate sample size (m >= 20 and n_test >= 20)
    is_statistically_validated = (m_cal >= 20) and (len(test_idx) >= 20)
    conformal_warnings = []
    if not is_statistically_validated:
        conformal_warnings.append(
            f"Small-sample conformal calibration (m_cal={m_cal}, n_test={len(test_idx)}). "
            f"Finite-sample guarantee provides conservative interval (width={mean_test_width:.2f} t/ha); "
            f"asymptotic coverage guarantees require larger multi-region parcel sets (n >= 50)."
        )

    # Save model weights & artifacts
    save_dir = os.path.join("models", exp_id)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), os.path.join(save_dir, "weights.pt"))
        
        # Register as active model in InferenceService
        from backend.app.models.inference import inference_service
        is_synth = True
        if "provenance" in labels_df.columns and len(labels_df["provenance"]) > 0:
            is_synth = "synthetic" in str(labels_df["provenance"].iloc[0])
        inference_service.save_active_model(
            model=model,
            model_type=model_type,
            feature_set=feature_set,
            feature_cols=feature_cols,
            y_mean=y_mean,
            y_std=y_std,
            val_residuals=val_residuals,
            empirical_coverage_pct=empirical_test_coverage,
            conformal_q95=conformal_q95,
            mean_interval_width=mean_test_width,
            cal_size=m_cal,
            test_size=len(test_idx),
            is_statistically_validated=is_statistically_validated,
            warnings=conformal_warnings,
            provenance="synthetic" if is_synth else "real_data",
            model_version=f"{model_type}_{feature_set}_v1"
        )
        
    predictions_map = {
        pid: round(float(y_pred_all[i]), 2)
        for i, pid in enumerate(common_pids)
    }
    
    result = {
        "experiment_id": exp_id,
        "model_type": model_type,
        "feature_set": feature_set,
        "features": feature_cols,
        "epochs": epochs,
        "duration_sec": duration_sec,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "conformal_uncertainty": {
            "cal_residuals_count": len(val_residuals),
            "test_samples_count": len(test_idx),
            "conformal_q95": round(conformal_q95, 3),
            "empirical_test_coverage_pct": round(empirical_test_coverage, 1),
            "mean_interval_width": round(mean_test_width, 2),
            "is_statistically_validated": is_statistically_validated,
            "warnings": conformal_warnings
        },
        "loss_history": loss_history,
        "split": {
            "train_count": len(train_pids),
            "val_count": len(val_pids),
            "test_count": len(test_pids)
        },
        "predictions": predictions_map,
        "save_dir": save_dir,
        "provenance": "synthetic_demonstration" if "synthetic" in labels_df.get("provenance", [""])[0] else "real_data"
    }
    
    # Persist experiment JSON
    os.makedirs("experiments", exist_ok=True)
    with open(os.path.join("experiments", f"{exp_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    return result


def ensure_baseline_model_trained(timeseries_df: pd.DataFrame, labels_df: pd.DataFrame, gdf: Any, force_retrain: bool = False):
    """
    Ensures an initial calibrated ST-GNN checkpoint exists in models/active_st_gnn
    with non-trivial trained weights and empirical validation residuals.
    """
    from backend.app.models.inference import ACTIVE_MODEL_DIR
    meta_path = os.path.join(ACTIVE_MODEL_DIR, "metadata.json")
    weights_path = os.path.join(ACTIVE_MODEL_DIR, "weights.pt")
    if not force_retrain and os.path.exists(meta_path) and os.path.exists(weights_path):
        return
        
    train_model_experiment(
        timeseries_df=timeseries_df,
        labels_df=labels_df,
        gdf=gdf,
        model_type="st_gnn",
        feature_set="fusion",
        epochs=35,
        lr=0.006,
        hidden_dim=32,
        dropout=0.08,
        random_seed=42,
        experiment_id="active_st_gnn"
    )
