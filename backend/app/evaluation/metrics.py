"""
CropYield-ST: Scientific Evaluation Metrics Module
Computes RMSE, MAE, R², MAPE, and residual distributions with robust handling of small sample sizes.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes standard regression benchmark metrics:
      - RMSE (tonnes/ha)
      - MAE (tonnes/ha)
      - R² (Coefficient of determination)
      - MAPE (% Mean Absolute Percentage Error)
      - Residuals summary
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    # Filter valid pairs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    n_samples = len(y_true)
    if n_samples == 0:
        return {
            "n_samples": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "mape": None,
            "status": "insufficient_data"
        }
        
    residuals = y_pred - y_true
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    
    # R-squared calculation with safety
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    
    if n_samples < 2 or ss_tot < 1e-9:
        r2 = 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))
        # Handle extreme negative R2 on very bad models
        r2 = max(-1.0, r2)
        
    # MAPE
    nonzero_mask = np.abs(y_true) > 1e-4
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100.0)
    else:
        mape = 0.0
        
    return {
        "n_samples": n_samples,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape_percent": round(mape, 2),
        "mean_actual": round(float(np.mean(y_true)), 3),
        "mean_predicted": round(float(np.mean(y_pred)), 3),
        "residual_mean": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4),
        "max_overprediction": round(float(np.max(residuals)), 4),
        "max_underprediction": round(float(np.min(residuals)), 4),
        "residuals": [round(float(r), 4) for r in residuals[:30]],
        "y_true": [round(float(yt), 2) for yt in y_true[:30]],
        "y_pred": [round(float(yp), 2) for yp in y_pred[:30]]
    }
