"""
Sentinel-1 Synthetic Aperture Radar (SAR) Acquisition and Preprocessing Pipeline

Scientific Context:
1. What SAR is:
   Synthetic Aperture Radar (SAR) is an active microwave remote sensing system. Unlike optical 
   sensors (Sentinel-2) that rely on reflected sunlight, Sentinel-1 emits coherent microwave 
   pulses (C-band, ~5.405 GHz, wavelength λ ≈ 5.6 cm) and measures the magnitude and phase of 
   the backscattered signal returning to the antenna.

2. Why Radar works through clouds:
   C-band microwave radiation (5.6 cm) is orders of magnitude larger than atmospheric aerosol 
   particles, water droplets, and cloud water vapor (< 0.01 cm). Consequently, Rayleigh 
   scattering is negligible, allowing Sentinel-1 pulses to penetrate clouds, haze, rain, and smoke 
   unimpeded, providing reliable imagery day, night, and throughout monsoon seasons.

3. Physical Meaning of VV and VH Polarizations:
   - VV (Vertical-Transmit, Vertical-Receive, co-polarization):
     The wave is transmitted and received in the vertical plane. Sensitive to direct surface 
     roughness, vertical stalk geometry, soil moisture, and ground dielectric properties.
   - VH (Vertical-Transmit, Horizontal-Receive, cross-polarization):
     The transmitted vertical wave undergoes multiple volumetric scatterings inside the dense 
     crop canopy (leaves, stems, branches), depolarizing the signal into the horizontal plane. 
     Strongly correlates with vegetative biomass, canopy volume, and structural complexity.
   - VV/VH Ratio (or VV_dB - VH_dB):
     Normalizes soil moisture fluctuations and provides a robust metric for canopy structural 
     development and crop phenology changes.
   - RVI (Radar Vegetation Index for Dual-Pol): RVI = 4 * VH / (VV + VH)
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def linear_to_db(linear_val: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Convert linear radar backscatter intensity (sigma nought) to decibels (dB).
    Formula: dB = 10 * log10(sigma0)
    """
    linear_val = np.asarray(linear_val, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(linear_val, eps))
    return db


def db_to_linear(db_val: np.ndarray) -> np.ndarray:
    """
    Convert radar backscatter from decibels (dB) back to linear intensity.
    Formula: sigma0 = 10^(dB / 10)
    """
    db_val = np.asarray(db_val, dtype=np.float32)
    return np.power(10.0, db_val / 10.0)


def compute_sar_features(
    vv_linear: np.ndarray,
    vh_linear: np.ndarray,
    eps: float = 1e-7
) -> Dict[str, np.ndarray]:
    """
    Calculate derived radar backscatter metrics and structural indices from VV and VH channels.
    
    Args:
        vv_linear: Co-polarization backscatter intensity (linear scale)
        vh_linear: Cross-polarization backscatter intensity (linear scale)
        eps: Small constant to prevent zero division
        
    Returns:
        Dict containing:
            - 'vv_linear', 'vh_linear'
            - 'vv_db', 'vh_db'
            - 'vv_vh_ratio' (linear ratio: VV / VH)
            - 'vv_minus_vh_db' (difference in dB: VV_dB - VH_dB)
            - 'rvi' (Radar Vegetation Index: 4*VH / (VV + VH))
    """
    vv_linear = np.asarray(vv_linear, dtype=np.float32)
    vh_linear = np.asarray(vh_linear, dtype=np.float32)
    
    vv_db = linear_to_db(vv_linear, eps=eps)
    vh_db = linear_to_db(vh_linear, eps=eps)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_vh_ratio = np.where(vh_linear > eps, vv_linear / vh_linear, np.nan)
        vv_minus_vh_db = vv_db - vh_db
        denom = vv_linear + vh_linear
        rvi = np.where(denom > eps, (4.0 * vh_linear) / denom, np.nan)
        
    # Physical clipping for safety
    rvi = np.clip(rvi, 0.0, 1.0)
    
    return {
        "vv_linear": vv_linear,
        "vh_linear": vh_linear,
        "vv_db": vv_db,
        "vh_db": vh_db,
        "vv_vh_ratio": vv_vh_ratio,
        "vv_minus_vh_db": vv_minus_vh_db,
        "rvi": rvi
    }


def generate_synthetic_sentinel1_sample(
    shape: Tuple[int, int] = (128, 128),
    seed: int = 42,
    base_field_map: Optional[np.ndarray] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate physically accurate C-band SAR backscatter arrays (VV, VH) corresponding to 
    agricultural parcel structures for reproducible testing and demonstration.
    
    Physical parameters modeled:
    - VV dB typical crop range: -15.0 dB to -6.0 dB (linear 0.03 to 0.25)
    - VH dB typical crop range: -24.0 dB to -12.0 dB (linear 0.004 to 0.063)
    - Radar Speckle: Multiplicative Gamma / Rayleigh distributed speckle noise (characteristic of coherent SAR)
    - Complete transparency to optical clouds
    """
    rng = np.random.RandomState(seed)
    h, w = shape
    
    if base_field_map is None:
        y_grid, x_grid = np.ogrid[:h, :w]
        num_fields = 16
        centers_x = rng.randint(10, w - 10, size=num_fields)
        centers_y = rng.randint(10, h - 10, size=num_fields)
        
        field_map = np.zeros((h, w), dtype=int)
        min_dist = np.full((h, w), np.inf)
        for i, (cx, cy) in enumerate(zip(centers_x, centers_y)):
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = dist < min_dist
            field_map[mask] = i
            min_dist[mask] = dist[mask]
    else:
        field_map = base_field_map
        num_fields = len(np.unique(field_map))
        
    # Assign canopy biomass / volume scattering vigor to each field (0.2 to 0.9)
    field_biomass = rng.uniform(0.3, 0.85, size=num_fields)
    pixel_biomass = field_biomass[field_map] + rng.normal(0, 0.02, size=(h, w))
    pixel_biomass = np.clip(pixel_biomass, 0.1, 0.95)
    
    # Model mean linear backscatter from physics
    # High biomass -> strong volume depolarization -> higher VH (-13 dB), moderately high VV (-8 dB)
    # Bare soil -> low volume depolarization -> low VH (-23 dB), moderate VV (-14 dB)
    vv_mean_db = -16.0 + 8.5 * pixel_biomass
    vh_mean_db = -24.0 + 11.5 * pixel_biomass
    
    vv_mean_linear = db_to_linear(vv_mean_db)
    vh_mean_linear = db_to_linear(vh_mean_db)
    
    # Simulate coherent SAR speckle noise (Gamma distribution with Equivalent Number of Looks ENL=4)
    enl = 4.0
    speckle_vv = rng.gamma(shape=enl, scale=1.0 / enl, size=(h, w))
    speckle_vh = rng.gamma(shape=enl, scale=1.0 / enl, size=(h, w))
    
    vv_linear = vv_mean_linear * speckle_vv
    vh_linear = vh_mean_linear * speckle_vh
    
    features = compute_sar_features(vv_linear, vh_linear)
    
    metadata = {
        "sensor": "Sentinel-1 C-SAR",
        "instrument_mode": "IW (Interferometric Wide Swath)",
        "product_type": "GRD (Ground Range Detected)",
        "frequency_ghz": 5.405,
        "wavelength_cm": 5.54,
        "spatial_resolution_m": 10.0,
        "polarizations": ["VV", "VH"],
        "num_fields": num_fields,
        "cloud_transparency": "100% (Microwave wavelength penetrates tropospheric clouds)",
        "enl_looks": enl
    }
    
    return features, metadata
