"""
Unit tests for Sentinel-1 Synthetic Aperture Radar (SAR) feature calculations.
"""
import numpy as np
import pytest
from backend.app.features.sar_features import (
    linear_to_db,
    db_to_linear,
    compute_vh_vv_ratio,
    compute_rvi_dual_pol,
    compute_all_sar_features
)


def test_linear_to_db_conversion():
    """Verify linear intensity conversion to decibels."""
    # sigma0 = 0.1 -> 10 * log10(0.1) = -10.0 dB
    # sigma0 = 0.01 -> -20.0 dB
    linear = np.array([0.1, 0.01, 1.0], dtype=np.float32)
    db = linear_to_db(linear)
    assert np.isclose(db[0], -10.0, atol=1e-4)
    assert np.isclose(db[1], -20.0, atol=1e-4)
    assert np.isclose(db[2], 0.0, atol=1e-4)


def test_db_to_linear_inversion():
    """Verify inverse roundtrip conversion."""
    db = np.array([-12.5, -18.2, -5.0], dtype=np.float32)
    linear = db_to_linear(db)
    recovered_db = linear_to_db(linear)
    assert np.allclose(db, recovered_db, atol=1e-4)


def test_vh_vv_ratio():
    """Verify cross-pol to co-pol ratio in linear and dB."""
    vv_lin = np.array([0.10, 0.20], dtype=np.float32)
    vh_lin = np.array([0.02, 0.05], dtype=np.float32)
    
    ratio_lin = compute_vh_vv_ratio(vv_lin, vh_lin, in_db=False)
    assert np.isclose(ratio_lin[0], 0.20, atol=1e-4)
    assert np.isclose(ratio_lin[1], 0.25, atol=1e-4)
    
    # In dB: VH_dB - VV_dB
    ratio_db = compute_vh_vv_ratio(np.array([-10.0]), np.array([-16.0]), in_db=True)
    assert np.isclose(ratio_db[0], -6.0, atol=1e-4)


def test_rvi_dual_pol():
    """Verify Radar Vegetation Index dual-pol range [0, 1]."""
    vv = np.array([0.10, 0.05], dtype=np.float32)
    vh = np.array([0.02, 0.025], dtype=np.float32)
    
    # RVI = 4 * VH / (VV + VH)
    # For index 0: 4 * 0.02 / (0.10 + 0.02) = 0.08 / 0.12 = 0.6667
    rvi = compute_rvi_dual_pol(vv, vh)
    assert np.isclose(rvi[0], 0.08 / 0.12, atol=1e-4)
    assert 0.0 <= rvi[0] <= 1.0
    assert 0.0 <= rvi[1] <= 1.0


def test_compute_all_sar_features():
    """Verify comprehensive dictionary output."""
    vv = np.array([0.08, 0.12], dtype=np.float32)
    vh = np.array([0.015, 0.025], dtype=np.float32)
    feats = compute_all_sar_features(vv, vh, is_db=False)
    
    for key in ["vv_db", "vh_db", "vv_linear", "vh_linear", "vh_vv_ratio", "rvi", "polarization_diff"]:
        assert key in feats
        assert len(feats[key]) == 2
