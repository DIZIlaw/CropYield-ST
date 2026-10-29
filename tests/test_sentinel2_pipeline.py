"""
Integration test for Sentinel-2 sample loader, index calculation, and visual output pipeline.
"""
import os
import yaml
import numpy as np
import pytest
from src.gee.sentinel2 import generate_synthetic_sentinel2_sample
from src.features.indices import compute_all_optical_indices
from src.visualization.maps import plot_rgb_composite, plot_single_index_map


def test_sentinel2_generation_and_shapes():
    """Verify Sentinel-2 band generator yields correct spatial dimensions and bands."""
    h, w = 64, 64
    bands, cloud_mask, meta = generate_synthetic_sentinel2_sample(shape=(h, w), seed=123, cloud_fraction=0.1)
    
    assert "B2" in bands
    assert "B3" in bands
    assert "B4" in bands
    assert "B5" in bands
    assert "B8" in bands
    assert "B11" in bands
    assert "SCL" in bands
    
    assert bands["B4"].shape == (h, w)
    assert cloud_mask.shape == (h, w)
    assert meta["width"] == w
    assert meta["height"] == h
    assert meta["cloud_pixel_count"] > 0


def test_end_to_end_phase1_processing(tmp_path):
    """Verify complete computation and figure rendering pipeline without errors."""
    bands, cloud_mask, meta = generate_synthetic_sentinel2_sample(shape=(32, 32), seed=42)
    indices = compute_all_optical_indices(bands, mask_cloud=cloud_mask)
    
    assert "ndvi" in indices
    assert "ndre" in indices
    assert "evi" in indices
    assert "ndwi" in indices
    
    # Check valid numeric ranges on unmasked pixels
    valid_ndvi = indices["ndvi"][~np.isnan(indices["ndvi"])]
    assert np.all(valid_ndvi >= -1.0)
    assert np.all(valid_ndvi <= 1.0)
    assert np.mean(valid_ndvi) > 0.1 # Agricultural parcels should have positive green vegetation signal
    
    # Check plot generation to temporary directory
    rgb_file = tmp_path / "rgb.png"
    ndvi_file = tmp_path / "ndvi.png"
    
    plot_rgb_composite(bands["B4"], bands["B3"], bands["B2"], output_path=str(rgb_file))
    plot_single_index_map(indices["ndvi"], "ndvi", output_path=str(ndvi_file))
    
    assert rgb_file.exists()
    assert ndvi_file.exists()
    assert rgb_file.stat().st_size > 0
    assert ndvi_file.stat().st_size > 0


def test_config_yaml_validity():
    """Verify config.yaml is well-formed and contains all required phase configuration fields."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    assert os.path.exists(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    assert "project" in config
    assert "study_area" in config
    assert "sentinel2" in config
    assert "sentinel1" in config
    assert "indices" in config
    assert config["study_area"]["crop"] == "Maize"
