"""
CropYield-ST: Synthetic Aperture Radar (SAR) Feature Engineering
Processes Sentinel-1 C-band dual-polarization (VV, VH) backscatter.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np


def linear_to_db(linear_val: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Convert linear radar backscatter intensity (sigma0) to decibels (dB).
    Formula: dB = 10 * log10(sigma0)
    """
    linear_val = np.asarray(linear_val, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        clipped = np.maximum(linear_val, eps)
        db = 10.0 * np.log10(clipped)
    return db


def db_to_linear(db_val: np.ndarray) -> np.ndarray:
    """
    Convert radar backscatter from decibels (dB) back to linear intensity.
    Formula: sigma0 = 10^(dB / 10)
    """
    db_val = np.asarray(db_val, dtype=np.float32)
    return np.power(10.0, db_val / 10.0)


def compute_vh_vv_ratio(
    vv: np.ndarray,
    vh: np.ndarray,
    in_db: bool = False,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Calculate VH/VV ratio (cross-pol to co-pol ratio).
    In linear scale: VH / VV
    In dB scale: VH_dB - VV_dB
    Physical significance: VH/VV rises as canopy structural complexity and volume increase,
    while canceling out common attenuation and soil dielectric changes.
    """
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    
    if in_db:
        # In dB, ratio is difference: 10*log10(VH) - 10*log10(VV)
        return vh - vv
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(vv) > eps, vh / (vv + eps), 0.0)
        return np.clip(ratio, 0.0, 5.0)


def compute_rvi_dual_pol(
    vv_linear: np.ndarray,
    vh_linear: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Radar Vegetation Index for Dual-Polarization (RVI_DP).
    Formula: RVI = 4 * VH / (VV + VH)
    Valid range: [0.0, 1.0] for typical vegetation.
    Physical significance: Directly correlates with plant canopy volume and aboveground fresh biomass.
    """
    vv = np.asarray(vv_linear, dtype=np.float32)
    vh = np.asarray(vh_linear, dtype=np.float32)
    
    denominator = vv + vh
    with np.errstate(divide="ignore", invalid="ignore"):
        rvi = np.where(denominator > eps, (4.0 * vh) / denominator, 0.0)
        
    return np.clip(rvi, 0.0, 1.0)


def compute_all_sar_features(
    vv: np.ndarray,
    vh: np.ndarray,
    is_db: bool = False,
    eps: float = 1e-7
) -> Dict[str, np.ndarray]:
    """
    Compute full suite of SAR features from VV and VH observations.
    Returns:
      - vv_db: VV in dB
      - vh_db: VH in dB
      - vv_linear: VV in linear
      - vh_linear: VH in linear
      - vh_vv_ratio: VH / VV linear ratio
      - rvi: Dual-polarization Radar Vegetation Index
      - polarization_diff: VV_dB - VH_dB (dB difference)
    """
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    
    if is_db:
        vv_db = vv
        vh_db = vh
        vv_linear = db_to_linear(vv_db)
        vh_linear = db_to_linear(vh_db)
    else:
        vv_linear = vv
        vh_linear = vh
        vv_db = linear_to_db(vv_linear, eps=eps)
        vh_db = linear_to_db(vh_linear, eps=eps)
        
    vh_vv_ratio = compute_vh_vv_ratio(vv_linear, vh_linear, in_db=False, eps=eps)
    rvi = compute_rvi_dual_pol(vv_linear, vh_linear, eps=eps)
    pol_diff = vv_db - vh_db
    
    return {
        "vv_db": vv_db,
        "vh_db": vh_db,
        "vv_linear": vv_linear,
        "vh_linear": vh_linear,
        "vh_vv_ratio": vh_vv_ratio,
        "rvi": rvi,
        "polarization_diff": pol_diff
    }
