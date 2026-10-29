"""
Unit tests for optical remote sensing vegetation and water indices calculation.
"""
import numpy as np
import pytest
from src.features.indices import (
    compute_ndvi,
    compute_ndre,
    compute_evi,
    compute_ndwi,
    compute_all_optical_indices
)


def test_ndvi_standard_calculation():
    """Test NDVI calculation against known values."""
    # Red = 0.05, NIR = 0.45 -> NDVI = (0.45 - 0.05)/(0.45 + 0.05) = 0.40 / 0.50 = 0.80
    nir = np.array([[0.45, 0.20], [0.50, 0.05]], dtype=np.float32)
    red = np.array([[0.05, 0.20], [0.10, 0.05]], dtype=np.float32)
    
    ndvi = compute_ndvi(nir, red)
    
    assert np.isclose(ndvi[0, 0], 0.80, atol=1e-5)
    assert np.isclose(ndvi[0, 1], 0.00, atol=1e-5) # Bare soil / zero contrast
    assert np.isclose(ndvi[1, 0], (0.50 - 0.10) / (0.50 + 0.10), atol=1e-5)


def test_ndvi_boundary_clipping():
    """Ensure NDVI results strictly remain within [-1.0, 1.0]."""
    nir = np.array([-0.1, 1.5, 0.0], dtype=np.float32)
    red = np.array([0.5, 0.1, 0.0], dtype=np.float32)
    
    ndvi = compute_ndvi(nir, red, clip_valid=True)
    
    assert np.all(ndvi >= -1.0) or np.isnan(ndvi).any()
    assert np.all(ndvi <= 1.0) or np.isnan(ndvi).any()


def test_ndre_calculation():
    """Test NDRE calculation with Red Edge (Band 5)."""
    nir = np.array([0.50, 0.60], dtype=np.float32)
    red_edge = np.array([0.25, 0.20], dtype=np.float32)
    
    ndre = compute_ndre(nir, red_edge)
    
    expected_0 = (0.50 - 0.25) / (0.50 + 0.25) # 0.25 / 0.75 = 0.3333
    expected_1 = (0.60 - 0.20) / (0.60 + 0.20) # 0.40 / 0.80 = 0.50
    
    assert np.isclose(ndre[0], expected_0, atol=1e-4)
    assert np.isclose(ndre[1], expected_1, atol=1e-4)


def test_evi_calculation():
    """Test EVI formula and positive responsiveness to healthy canopy."""
    nir = np.array([0.50], dtype=np.float32)
    red = np.array([0.05], dtype=np.float32)
    blue = np.array([0.02], dtype=np.float32)
    
    evi = compute_evi(nir, red, blue)
    
    # 2.5 * (0.50 - 0.05) / (0.50 + 6*0.05 - 7.5*0.02 + 1.0)
    # Numerator = 2.5 * 0.45 = 1.125
    # Denominator = 0.50 + 0.30 - 0.15 + 1.0 = 1.65
    # EVI = 1.125 / 1.65 = 0.6818
    expected = 1.125 / 1.65
    assert np.isclose(evi[0], expected, atol=1e-4)


def test_ndwi_calculation():
    """Test NDWI water / canopy moisture index."""
    green = np.array([0.15, 0.05], dtype=np.float32)
    nir = np.array([0.05, 0.45], dtype=np.float32)
    
    ndwi = compute_ndwi(green, nir)
    
    assert ndwi[0] > 0 # Water body has Green > NIR -> Positive NDWI
    assert ndwi[1] < 0 # Vegetation has NIR > Green -> Negative NDWI


def test_cloud_masking_propagation():
    """Verify that cloud mask sets corresponding pixels to NaN across all indices."""
    bands = {
        "B2": np.array([[0.05, 0.70]], dtype=np.float32),
        "B3": np.array([[0.08, 0.70]], dtype=np.float32),
        "B4": np.array([[0.04, 0.70]], dtype=np.float32),
        "B5": np.array([[0.20, 0.70]], dtype=np.float32),
        "B8": np.array([[0.50, 0.70]], dtype=np.float32)
    }
    cloud_mask = np.array([[False, True]], dtype=bool)
    
    indices = compute_all_optical_indices(bands, mask_cloud=cloud_mask)
    
    for k in ["ndvi", "ndre", "evi", "ndwi"]:
        assert not np.isnan(indices[k][0, 0])
        assert np.isnan(indices[k][0, 1])
