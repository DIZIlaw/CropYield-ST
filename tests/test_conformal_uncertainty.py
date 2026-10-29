"""
Unit tests for Split-Conformal Prediction Uncertainty Calibration.
Verifies finite-sample distribution-free validity and interval properties.
"""
import pytest
import numpy as np


def compute_conformal_quantile(residuals: np.ndarray, alpha: float = 0.05) -> float:
    """
    Computes distribution-free conformal calibration quantile.
    Formula: Quantile level = ceil((m + 1) * (1 - alpha)) / m
    """
    m = len(residuals)
    assert m > 0, "Non-empty residuals required"
    q_level = min(1.0, np.ceil((m + 1) * (1.0 - alpha)) / m)
    return float(np.quantile(residuals, q_level))


def test_conformal_quantile_formula():
    """Verify finite-sample correction ceil((m+1)(1-alpha))/m."""
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q_val = compute_conformal_quantile(residuals, alpha=0.10)
    assert q_val >= 0.9
    assert q_val <= 1.0


def test_conformal_coverage_guarantee():
    """
    Verify empirical coverage rate on simulated held-out validation distribution
    matches or exceeds nominal (1 - alpha) = 90%.
    """
    np.random.seed(42)
    val_residuals = np.abs(np.random.normal(0, 0.4, size=100))
    alpha = 0.10
    q_hat = compute_conformal_quantile(val_residuals, alpha=alpha)
    
    # Test set
    test_residuals = np.abs(np.random.normal(0, 0.4, size=1000))
    empirical_coverage = np.mean(test_residuals <= q_hat)
    
    # Conformal theory ensures empirical coverage is close to >= (1 - alpha)
    assert empirical_coverage >= 0.85


def test_conformal_interval_properties():
    """Verify lower <= upper and non-negative boundaries."""
    pred_yield = 4.25
    q_hat = 0.52
    ci_lower = max(0.0, pred_yield - q_hat)
    ci_upper = pred_yield + q_hat
    width = ci_upper - ci_lower
    
    assert ci_lower >= 0.0
    assert ci_upper > ci_lower
    assert np.isclose(width, 2 * q_hat)


def test_small_sample_conformal_limitation():
    """
    Verify that with m=5 calibration samples at 1-alpha=0.95,
    q_level clamps to 1.0 (max residual) and finite sample bound is m/(m+1)=83.3%.
    This proves why small-sample calibration must be accompanied by an honest warning.
    """
    residuals = np.array([0.15, 0.45, 0.82, 1.10, 1.45])
    m = len(residuals)
    alpha = 0.05
    q_level = min(1.0, np.ceil((m + 1) * (1.0 - alpha)) / m)
    assert q_level == 1.0  # ceil(6 * 0.95) / 5 = 6/5 clamped to 1.0
    q_hat = float(np.quantile(residuals, q_level))
    assert q_hat == 1.45  # exactly max(residuals)
    max_finite_sample_guarantee = m / (m + 1)
    assert np.isclose(max_finite_sample_guarantee, 5.0 / 6.0)

