"""
Unit tests for regression metrics (RMSE, MAE, R², MAPE) and edge case handling.
"""
import numpy as np
import pytest
from backend.app.evaluation.metrics import compute_regression_metrics


def test_perfect_predictions():
    """Verify that perfect predictions yield 0 RMSE and 1.0 R²."""
    y = np.array([3.5, 4.2, 5.1, 6.0])
    res = compute_regression_metrics(y, y)
    
    assert res["rmse"] == 0.0
    assert res["mae"] == 0.0
    assert res["r2"] == 1.0
    assert res["mape_percent"] == 0.0


def test_known_error_metrics():
    """Verify metrics against hand-calculated values."""
    y_true = np.array([4.0, 5.0, 6.0])
    y_pred = np.array([4.5, 4.5, 7.0])
    # Residuals: [0.5, -0.5, 1.0]
    # MSE = (0.25 + 0.25 + 1.0) / 3 = 1.5 / 3 = 0.5
    # RMSE = sqrt(0.5) ≈ 0.7071
    # MAE = (0.5 + 0.5 + 1.0) / 3 = 2.0 / 3 ≈ 0.6667
    
    res = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(res["rmse"], np.sqrt(0.5), atol=1e-3)
    assert np.isclose(res["mae"], 2.0 / 3.0, atol=1e-3)


def test_empty_or_nan_handling():
    """Verify graceful handling when data has NaNs or is empty."""
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([1.0, 2.0])
    res = compute_regression_metrics(y_true, y_pred)
    assert res["status"] == "insufficient_data"
    assert res["rmse"] is None
